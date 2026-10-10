"""git-merger: Semantic Merge-Conflict Resolver with Differential Verification."""

from git_merger.ast_analysis import (
    BlastRadiusAnalyzer,
    build_call_graph,
    diff_functions,
    extract_functions,
    get_function_source,
    get_signature,
)
from git_merger.evaluator import DifferentialEvaluator
from git_merger.exceptions import (
    ASTAnalysisError,
    GitError,
    GitMergerError,
    IngestionError,
)
from git_merger.ingester import ContextIngester
from git_merger.models import (
    AgentState,
    BlastRadius,
    BranchContext,
    EvaluationVerdict,
    ExecutionResult,
    FileVersion,
    FunctionChange,
    MergeContext,
    ParallelExecutionResult,
    SandboxSpec,
    SandboxStatus,
    TestHarness,
)
from git_merger.pytest_parser import parse_pytest_output
from git_merger.sandbox import (
    MockSandboxProvider,
    SandboxProvider,
    get_sandbox_provider,
)
from git_merger.sandbox_manager import SandboxManager

__all__ = [
    "ContextIngester",
    "BlastRadiusAnalyzer",
    "extract_functions",
    "get_function_source",
    "get_signature",
    "diff_functions",
    "build_call_graph",
    "FileVersion",
    "FunctionChange",
    "BranchContext",
    "MergeContext",
    "BlastRadius",
    "AgentState",
    "SandboxStatus",
    "SandboxSpec",
    "TestHarness",
    "ExecutionResult",
    "ParallelExecutionResult",
    "EvaluationVerdict",
    "SandboxProvider",
    "MockSandboxProvider",
    "get_sandbox_provider",
    "SandboxManager",
    "DifferentialEvaluator",
    "parse_pytest_output",
    "GitMergerError",
    "GitError",
    "IngestionError",
    "ASTAnalysisError",
]
