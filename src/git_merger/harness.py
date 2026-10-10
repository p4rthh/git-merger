
import json
from dataclasses import dataclass, field

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from git_merger.models import MergeContext


@dataclass
class TestHarness:
    test_code: str
    conftest_code: str
    target_functions: list[str]
    test_count: int
    pip_requirements: list[str]
    hypothesis_strategies: dict[str, str] = field(default_factory=dict)


class HarnessResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_code: str
    conftest_code: str
    target_functions: list[str]
    test_count: int = Field(..., ge=1)
    pip_requirements: list[str]
    hypothesis_strategies: dict[str, str] = Field(
        default_factory=dict
    )


SYSTEM_PROMPT = """
You are an expert Python test engineer specializing in property-based
testing and differential verification of semantic merge resolutions.

You are given:
1. The BASE version of a Python file.
2. Branch A (OURS).
3. Branch B (THEIRS).
4. The synthesized merge candidate.
5. A list of functions affected by the changes.

Your task is to generate a self-contained pytest test harness that
checks whether the candidate preserves the behavioral changes from
both branches without introducing regressions.

CRITICAL RULES:
1. Tests must be SELF-CONTAINED — no external dependencies beyond
   pytest and hypothesis.
2. Each test function must have a clear docstring explaining the
   invariant it checks.
3. Generate at least 3 test functions per changed function.
4. Include both positive tests and negative tests for error handling.
5. DO NOT test unchanged functions.
6. Use this exact dynamic module-loading pattern:

   import importlib
   import os
   MODULE_NAME = os.environ.get("TARGET_MODULE", "candidate_version")
   mod = importlib.import_module(MODULE_NAME)

7. Compare behavior against the base and both branches where
   appropriate. Do not assume the candidate is correct simply
   because it compiles.
8. Do not invent APIs or expected behavior unsupported by the
   supplied code. If behavior is ambiguous, write a test only when
   the expected invariant can be justified from the inputs.
9. Return complete, runnable pytest test code.
10. Respond with a JSON object matching the requested schema exactly.
    Do not include markdown fences or text outside the JSON object.

The JSON must contain:
- test_code
- conftest_code
- target_functions
- test_count
- pip_requirements
- hypothesis_strategies
"""


USER_PROMPT_TEMPLATE = """
## DIFFERENTIAL TEST HARNESS GENERATION

### FILE
{file_path}

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

### MERGED CANDIDATE
```python
{candidate_code}
```

### FUNCTIONS MODIFIED BY BOTH BRANCHES
{shared_changed_functions}

### TARGET FUNCTIONS TO TEST
{target_functions}

### AST CHANGE SUMMARY
{ast_diff_summary_json}

Generate a differential pytest harness that verifies the changed
functions and checks for regressions.

Return a JSON object with exactly these fields:
{{
  "test_code": "Complete pytest test file as a string",
  "conftest_code": "conftest.py contents, or an empty string",
  "target_functions": ["function_name"],
  "test_count": 3,
  "pip_requirements": [],
  "hypothesis_strategies": {{
    "function_name": "Description of generated test inputs"
  }}
}}
"""


class HarnessGenerator:
    def __init__(self, model_client: OpenAI):
        self.client = model_client
        self.synthesis_model = "nvidia/nemotron-3-super-120b-a12b"
        self.strategy_model = "nvidia/nemotron-3-nano-30b-a3b"
        self.last_token_usage: dict[str, int] = {}

    def generate(
        self,
        context: MergeContext,
        candidate_code: str,
        target_functions: list[str] | None = None,
    ) -> TestHarness:
        """
        Generate a differential test harness for the merge candidate.

        If target_functions is provided, only those functions are
        requested for testing. This allows BlastRadius.affected to
        limit testing to functions affected by the changes.
        """

        if target_functions is None:
            target_functions = sorted(
                set(context.shared_changed_functions)
            )

        if not target_functions:
            raise ValueError(
                "No target functions were provided for harness generation."
            )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            file_path=context.file_path,
            base_content=context.base.content,
            ours_content=context.ours.file_version.content,
            theirs_content=context.theirs.file_version.content,
            candidate_code=candidate_code,
            shared_changed_functions=", ".join(
                context.shared_changed_functions
            ),
            target_functions=", ".join(target_functions),
            ast_diff_summary_json=json.dumps(
                context.ast_diff_summary,
                indent=2,
                default=str,
            ),
        )

        response = self.client.chat.completions.create(
            model=self.synthesis_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=8192,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content

        if content is None:
            raise ValueError(
                "Nemotron returned an empty harness response."
            )

        parsed = HarnessResponse.model_validate_json(content)

        usage = response.usage
        self.last_token_usage = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": (
                usage.completion_tokens if usage else 0
            ),
        }

        return TestHarness(
            test_code=parsed.test_code,
            conftest_code=parsed.conftest_code,
            target_functions=parsed.target_functions,
            test_count=parsed.test_count,
            pip_requirements=parsed.pip_requirements,
            hypothesis_strategies=parsed.hypothesis_strategies,
        )