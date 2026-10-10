
import json

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from git_merger.models import MergeContext, SynthesisResult
from git_merger.ingester import ContextIngester


class ProvenanceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(
        ...,
        pattern="^(ours|theirs|both|base|novel)$",
    )
    rationale: str
    changes_applied: list[str]


class SynthesisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_code: str
    explanation: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    per_function_provenance: dict[str, ProvenanceInfo]


SYSTEM_PROMPT = """You are a Principal Software Engineer specializing in semantic merge conflict resolution for Python codebases. You are given the full 3-way merge context for a Python file: the common ancestor (BASE), the changes from Branch A (OURS), and the changes from Branch B (THEIRS).

Your task is to produce a SINGLE unified Python source file that correctly incorporates the intent of BOTH branches without introducing regressions, dropping functionality, or creating syntax errors.

CRITICAL RULES:
1. NEVER silently drop code from either branch. If Branch A added a nil-check, it MUST appear in your output. If Branch B changed a function signature, the new signature MUST be used.
2. When both branches modify the same function, you must MERGE their changes, not pick one side.
3. If Branch A changed a function signature, ALL call sites and ALL body logic from Branch B must be adapted to the new signature.
4. Preserve ALL imports, ALL module-level constants, and ALL class definitions from both branches.
5. Your output must be syntactically valid Python 3.11+.
6. Explain your reasoning for every merge decision at the function level.

You MUST respond with a JSON object matching the provided schema exactly. Do not include markdown fences or any text outside the JSON.
"""


USER_PROMPT_TEMPLATE = """## MERGE CONTEXT

### File: {file_path}

### COMMIT HISTORY (Branch A / OURS):
{ours_commit_log}

### COMMIT HISTORY (Branch B / THEIRS):
{theirs_commit_log}

### BASE VERSION (Common Ancestor):
```python
{base_content}
```

### BRANCH A (OURS) — Full File:
```python
{ours_content}
```

### BRANCH B (THEIRS) — Full File:
```python
{theirs_content}
```

### DIFF: BASE → BRANCH A
```diff
{diff_base_to_ours}
```

### DIFF: BASE → BRANCH B
```diff
{diff_base_to_theirs}
```

### AST CHANGE SUMMARY:
{ast_diff_summary_json}

### FUNCTIONS MODIFIED BY BOTH BRANCHES:
{shared_changed_functions}

Produce the unified merged Python file and per-function provenance.
"""


class SemanticSynthesizer:
    def __init__(self, model_client: OpenAI):
        self.client = model_client
        self.model = "nvidia/nemotron-3-ultra-550b-a55b"
        self.last_token_usage: dict[str, int] = {}

    def synthesize(self, context: MergeContext) -> SynthesisResult:
        """
        Generate an initial merged candidate using the BASE, OURS,
        and THEIRS versions of the file.
        """

        user_prompt = USER_PROMPT_TEMPLATE.format(
            file_path=context.file_path,
            ours_commit_log="\n".join(context.ours.commit_log),
            theirs_commit_log="\n".join(context.theirs.commit_log),
            base_content=context.base.content,
            ours_content=context.ours.file_version.content,
            theirs_content=context.theirs.file_version.content,
            diff_base_to_ours=context.ours.diff_from_base,
            diff_base_to_theirs=context.theirs.diff_from_base,
            ast_diff_summary_json=json.dumps(
                context.ast_diff_summary,
                indent=2,
                default=str,
            ),
            shared_changed_functions=", ".join(
                context.shared_changed_functions
            ),
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.1,
            max_tokens=8192,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content

        if content is None:
            raise ValueError(
                "Nemotron returned an empty synthesis response."
            )

        # Validate the returned JSON against the expected schema.
        parsed = SynthesisResponse.model_validate_json(content)

        # Record token usage for the project's token budget.
        usage = response.usage

        self.last_token_usage = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": (
                usage.completion_tokens if usage else 0
            ),
        }

        # Convert the validated response into the project's dataclass.
        return SynthesisResult(
            candidate_code=parsed.candidate_code,
            explanation=parsed.explanation,
            confidence=parsed.confidence,
            per_function_provenance={
                function_name: provenance.model_dump()
                for function_name, provenance
                in parsed.per_function_provenance.items()
            },
            model_used=self.model,
            token_usage=self.last_token_usage.copy(),
        )


if __name__ == "__main__":
    ingester = ContextIngester(repo_path="path/to/repo")
    synthesizer = SemanticSynthesizer(model_client=OpenAI())
    context = ingester.ingest(
        branch_a="branch-a",
        branch_b="branch-b",
        file_path="path/to/file.py",
    )
    result = synthesizer.synthesize(context=context)

    print(result.candidate_code)
    print(result.explanation)
    print(result.confidence)