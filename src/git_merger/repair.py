
import json
from dataclasses import asdict

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from git_merger.models import MergeContext, EvaluationVerdict


class FixApplied(BaseModel):
    failure_id: str
    fix_description: str


class RepairResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_code: str
    fixes_applied: list[FixApplied]
    confidence: float = Field(..., ge=0.0, le=1.0)
    root_cause_analysis: str
    is_genuine_conflict: bool = False
    escalation_reason: str = ""


SYSTEM_PROMPT = """
You are a Principal Software Engineer performing iterative repair
on a failed merge resolution. A previous merge candidate M* was
generated but FAILED differential testing against the original branches.

You are given:
1. The original merge context (BASE, OURS, THEIRS).
2. The PREVIOUS CANDIDATE that failed.
3. FAILURE DETAILS: which tests failed, expected vs actual output,
   and stderr/stdout traces.
4. The REPAIR ITERATION NUMBER. You have at most {max_iterations} attempts.

Your task is to produce a CORRECTED candidate that fixes ALL reported
failures while maintaining ALL previously correct behaviors.

CRITICAL RULES:
1. Analyze each failure trace carefully. Identify the ROOT CAUSE
   (missing nil-check, wrong signature, dropped logic, etc.).
2. Do NOT make unnecessary changes. Only modify the minimal code
   needed to fix the failures.
3. Do NOT drop previously working functionality to fix a new failure.
4. If a failure indicates a genuine semantic conflict (both branches
   changed the SAME behavior in incompatible ways), explain this in
   your response and mark confidence below 0.3.
5. Each repair iteration must make PROGRESS by fixing at least one
   failure that was present before.

You MUST respond with a JSON object matching the provided schema exactly.
Do not include markdown fences or text outside the JSON object.
"""


USER_PROMPT_TEMPLATE = """
## REPAIR ITERATION {iteration} of {max_iterations}

### ORIGINAL MERGE CONTEXT
- File: {file_path}
- Functions modified by both branches: {shared_changed_functions}

### BASE VERSION
```python
{base_content}
```

### BRANCH A (OURS)
```python
{ours_content}
```

### BRANCH B (THEIRS)
```python
{theirs_content}
```

### PREVIOUS CANDIDATE (FAILED)
```python
{previous_candidate}
```

### FAILURE REPORT
{failure_details_json}

### TEST HARNESS (FOR REFERENCE)
```python
{test_harness_code}
```

### EXECUTION OUTPUTS

#### Candidate M* stdout
```
{candidate_stdout}
```

#### Candidate M* stderr
```
{candidate_stderr}
```

#### Branch A stdout (reference)
```
{ours_stdout}
```

#### Branch B stdout (reference)
```
{theirs_stdout}
```

Produce a corrected candidate that fixes all reported failures while
preserving previously working behavior.

Return exactly these JSON fields:
- candidate_code: corrected merged Python source
- fixes_applied: list of objects containing failure_id and fix_description
- confidence: number between 0.0 and 1.0
- root_cause_analysis: explanation of the underlying failures
- is_genuine_conflict: boolean
- escalation_reason: explanation if human review is needed, otherwise ""

If a genuine semantic conflict exists, set confidence below 0.3 and
explain why it cannot safely be resolved automatically.
"""


class RepairSynthesizer:
    def __init__(self, model_client: OpenAI):
        self.client = model_client
        self.model = "nvidia/nemotron-3-ultra-550b-a55b"
        self.last_token_usage: dict[str, int] = {}

    def repair(
        self,
        context: MergeContext,
        previous_candidate: str,
        verdict: EvaluationVerdict,
        iteration: int,
        max_iterations: int,
        test_harness_code: str = "",
    ) -> dict:
        """
        Repair a failed merge candidate using Nemotron Ultra.

        Returns a dictionary containing the corrected candidate,
        applied fixes, confidence, root-cause analysis, conflict
        status, escalation reason, and token usage.
        """

        if iteration < 1:
            raise ValueError("iteration must be at least 1.")

        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")

        if iteration > max_iterations:
            raise ValueError(
                "iteration cannot exceed max_iterations."
            )

        failure_details_json = json.dumps(
            asdict(verdict),
            indent=2,
            default=str,
        )

        candidate_stdout = verdict.raw_outputs.candidate_result.stdout
        candidate_stderr = verdict.raw_outputs.candidate_result.stderr
        ours_stdout = verdict.raw_outputs.ours_result.stdout
        theirs_stdout = verdict.raw_outputs.theirs_result.stdout

        user_prompt = USER_PROMPT_TEMPLATE.format(
            iteration=iteration,
            max_iterations=max_iterations,
            file_path=context.file_path,
            shared_changed_functions=", ".join(
                context.shared_changed_functions
            ),
            base_content=context.base.content,
            ours_content=context.ours.file_version.content,
            theirs_content=context.theirs.file_version.content,
            previous_candidate=previous_candidate,
            failure_details_json=failure_details_json,
            test_harness_code=test_harness_code,
            candidate_stdout=candidate_stdout,
            candidate_stderr=candidate_stderr,
            ours_stdout=ours_stdout,
            theirs_stdout=theirs_stdout,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(
                        max_iterations=max_iterations
                    ),
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.15,
            max_tokens=8192,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content

        if content is None:
            raise ValueError(
                "Nemotron returned an empty repair response."
            )

        parsed = RepairResponse.model_validate_json(content)

        usage = response.usage
        self.last_token_usage = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": (
                usage.completion_tokens if usage else 0
            ),
        }

        return {
            "candidate_code": parsed.candidate_code,
            "fixes_applied": [
                fix.model_dump() for fix in parsed.fixes_applied
            ],
            "confidence": parsed.confidence,
            "root_cause_analysis": parsed.root_cause_analysis,
            "is_genuine_conflict": parsed.is_genuine_conflict,
            "escalation_reason": parsed.escalation_reason,
            "token_usage": self.last_token_usage.copy(),
        }