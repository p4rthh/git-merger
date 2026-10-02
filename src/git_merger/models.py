from dataclasses import dataclass, field
from enum import Enum, auto


@dataclass
class FileVersion:
    """A single version of a file at a specific point in history."""

    content: str
    commit_hash: str
    commit_message: str
    author: str
    timestamp: str


@dataclass
class FunctionChange:
    """Metadata about a changed function."""

    name: str
    change_type: str
    base_source: str | None
    branch_source: str | None
    signature_changed: bool
    body_changed: bool


@dataclass
class BranchContext:
    """Full context for one side of the merge."""

    file_version: FileVersion
    commit_log: list[str]
    diff_from_base: str
    changed_functions: list[FunctionChange]
    full_diff_patch: str


@dataclass
class MergeContext:
    """Complete three-way merge context for a single file."""

    file_path: str
    base: FileVersion
    ours: BranchContext
    theirs: BranchContext
    conflict_markers: str | None
    ast_diff_summary: dict[str, object]
    shared_changed_functions: list[str]


@dataclass
class SynthesisResult:
    """Output of the semantic synthesis step."""

    candidate_code: str
    explanation: str
    confidence: float
    per_function_provenance: dict[str, dict[str, str]]
    model_used: str
    token_usage: dict[str, int]


@dataclass
class TestHarness:
    """A differential test harness ready for sandbox execution."""

    test_code: str
    conftest_code: str
    requirements: list[str]
    target_functions: list[str]
    estimated_runtime_seconds: int
    test_count: int


@dataclass
class ExecutionResult:
    """Result of executing a command inside a sandbox."""

    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool


@dataclass
class SandboxSpec:
    """Configuration used to provision a sandbox."""

    python_version: str = "3.11"
    memory_mb: int = 512
    timeout_seconds: int = 60
    pip_packages: list[str] = field(default_factory=lambda: ["pytest", "hypothesis"])


@dataclass
class SandboxFileTree:
    """Files to inject into a sandbox."""

    files: dict[str, str]


@dataclass
class ParallelExecutionResult:
    """Results from running the harness against all four versions."""

    base_result: ExecutionResult
    ours_result: ExecutionResult
    theirs_result: ExecutionResult
    candidate_result: ExecutionResult
    all_passed: bool


@dataclass
class EvaluationVerdict:
    """Result of differential evaluation."""

    passed: bool
    failures: list[dict[str, object]]
    summary: str
    branch_a_preserved: bool
    branch_b_preserved: bool
    regression_detected: bool
    raw_outputs: ParallelExecutionResult


@dataclass
class RepairAttempt:
    """Record of a single repair iteration."""

    iteration: int
    previous_candidate: str
    failure_report: EvaluationVerdict
    new_candidate: str
    new_explanation: str
    token_usage: dict[str, int]


class AgentState(Enum):
    """Current stage of the merge-resolution workflow."""

    IDLE = auto()
    INGESTING = auto()
    SYNTHESIZING = auto()
    GENERATING_HARNESS = auto()
    SANDBOX_EVALUATING = auto()
    REPAIRING = auto()
    SUCCESS = auto()
    COMMITTED = auto()
    ESCALATING_TO_HUMAN = auto()


@dataclass
class TokenBudget:
    """Track and enforce token usage limits."""

    max_total_tokens: int = 500_000
    prompt_tokens_used: int = 0
    completion_tokens_used: int = 0

    @property
    def total_used(self) -> int:
        return self.prompt_tokens_used + self.completion_tokens_used

    def has_remaining(self) -> bool:
        return self.total_used < self.max_total_tokens

    def record(self, usage: dict[str, int]) -> None:
        self.prompt_tokens_used += usage.get("prompt_tokens", 0)
        self.completion_tokens_used += usage.get("completion_tokens", 0)

    def summary(self) -> dict[str, int | float]:
        utilization_pct = (
            round(self.total_used / self.max_total_tokens * 100, 1)
            if self.max_total_tokens
            else 0.0
        )
        return {
            "prompt_tokens": self.prompt_tokens_used,
            "completion_tokens": self.completion_tokens_used,
            "total_tokens": self.total_used,
            "budget_remaining": self.max_total_tokens - self.total_used,
            "utilization_pct": utilization_pct,
        }


@dataclass
class ModelRoute:
    """Configuration for routing a request to a model."""

    model_id: str
    max_tokens: int
    temperature: float
    json_mode: bool


MODEL_ROUTES: dict[str, ModelRoute] = {
    "intent_synthesis": ModelRoute(
        model_id="nvidia/nemotron-3-ultra-550b-a55b",
        max_tokens=8192,
        temperature=0.1,
        json_mode=True,
    ),
    "harness_generation": ModelRoute(
        model_id="nvidia/nemotron-3-super-120b-a12b",
        max_tokens=8192,
        temperature=0.2,
        json_mode=True,
    ),
    "log_parsing": ModelRoute(
        model_id="nvidia/nemotron-3-nano-30b-a3b",
        max_tokens=4096,
        temperature=0.0,
        json_mode=True,
    ),
    "repair": ModelRoute(
        model_id="nvidia/nemotron-3-ultra-550b-a55b",
        max_tokens=8192,
        temperature=0.15,
        json_mode=True,
    ),
    "strategy_selection": ModelRoute(
        model_id="nvidia/nemotron-3-nano-30b-a3b",
        max_tokens=2048,
        temperature=0.0,
        json_mode=True,
    ),
}