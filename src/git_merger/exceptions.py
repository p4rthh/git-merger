"""Custom exceptions for git-merger."""


class GitMergerError(Exception):
    """Base exception for all git-merger errors."""
    pass


class GitError(GitMergerError):
    """Raised when a Git command fails or a repository operation is invalid."""
    def __init__(self, message: str, returncode: int | None = None, stderr: str | None = None):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


class IngestionError(GitMergerError):
    """Raised when context ingestion fails for a merge candidate."""
    pass


class ASTAnalysisError(GitMergerError):
    """Raised when AST parsing or analysis fails."""
    pass
