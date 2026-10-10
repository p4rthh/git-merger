"""git-merger: Semantic Merge-Conflict Resolver with Differential Verification."""

from git_merger.ast_analysis import (
    BlastRadiusAnalyzer,
    build_call_graph,
    diff_functions,
    extract_functions,
    get_function_source,
    get_signature,
)
from git_merger.exceptions import (
    ASTAnalysisError,
    GitError,
    GitMergerError,
    IngestionError,
)
from git_merger.ingester import ContextIngester
from git_merger.models import (
    BlastRadius,
    BranchContext,
    FileVersion,
    FunctionChange,
    MergeContext,
    SynthesisResult,
)
from git_merger.synthesizer import SemanticSynthesizer

__all__ = [
    "ContextIngester",
    "SemanticSynthesizer",
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
    "SynthesisResult",
    "BlastRadius",
    "GitMergerError",
    "GitError",
    "IngestionError",
    "ASTAnalysisError",
]
