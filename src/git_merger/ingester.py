"""Context Ingestion engine for git-merger (Section 3.2.1)."""

import os
from typing import Optional

from git_merger import git_ops
from git_merger.ast_analysis import (
    BlastRadiusAnalyzer,
    analyze_function_change,
    extract_functions,
)
from git_merger.exceptions import GitError, IngestionError
from git_merger.models import (
    BlastRadius,
    BranchContext,
    FileVersion,
    FunctionChange,
    MergeContext,
)


class ContextIngester:
    """
    Ingests and normalizes the full 3-way merge context from a Git repository
    into structured MergeContext objects. Follows Section 3.2.1 of PROJECT_SPEC.md.
    """

    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        self.blast_analyzer = BlastRadiusAnalyzer()

    def ingest(
        self,
        branch_a: str,
        branch_b: str,
        file_path: str,
    ) -> MergeContext:
        """
        Extract full merge context for a conflicting file.

        Args:
            branch_a: Name or SHA of "ours" branch.
            branch_b: Name or SHA of "theirs" branch.
            file_path: Path to the conflicting file (relative to repo root).

        Returns:
            MergeContext with all three versions, diffs, commit history,
            and AST-level change analysis.
        """
        try:
            base_sha = self._compute_merge_base(branch_a, branch_b)
        except Exception as e:
            raise IngestionError(
                f"Failed to compute merge base between '{branch_a}' and '{branch_b}': {e}"
            ) from e

        try:
            base_version = self._extract_file_version(base_sha, file_path)
            ours_version = self._extract_file_version(branch_a, file_path)
            theirs_version = self._extract_file_version(branch_b, file_path)
        except Exception as e:
            raise IngestionError(
                f"Failed to extract file revisions for '{file_path}': {e}"
            ) from e

        # Extract commit history logs
        try:
            ours_commits = git_ops.get_commit_log(self.repo_path, base_sha, branch_a, file_path)
            theirs_commits = git_ops.get_commit_log(self.repo_path, base_sha, branch_b, file_path)
        except GitError as e:
            raise IngestionError(f"Failed to extract commit logs for '{file_path}': {e}") from e

        # Extract unified diffs
        try:
            diff_ours = git_ops.get_diff(self.repo_path, base_sha, branch_a, file_path)
            diff_theirs = git_ops.get_diff(self.repo_path, base_sha, branch_b, file_path)
        except GitError as e:
            raise IngestionError(f"Failed to extract diffs for '{file_path}': {e}") from e

        # Perform AST-level change analysis
        ours_changed_fns = self._analyze_ast_changes(base_version.content, ours_version.content)
        theirs_changed_fns = self._analyze_ast_changes(base_version.content, theirs_version.content)

        ours_context = BranchContext(
            file_version=ours_version,
            commit_log=ours_commits,
            diff_from_base=diff_ours,
            changed_functions=ours_changed_fns,
            full_diff_patch=diff_ours,
        )

        theirs_context = BranchContext(
            file_version=theirs_version,
            commit_log=theirs_commits,
            diff_from_base=diff_theirs,
            changed_functions=theirs_changed_fns,
            full_diff_patch=diff_theirs,
        )

        # Check for conflict markers in the working tree copy
        conflict_markers = None
        working_content = git_ops.read_working_tree_file(self.repo_path, file_path)
        if working_content and ("<<<<<<<" in working_content and ">>>>>>>" in working_content):
            conflict_markers = working_content

        # Shared changed functions (functions modified by BOTH branches)
        ours_names = {fc.name for fc in ours_changed_fns}
        theirs_names = {fc.name for fc in theirs_changed_fns}
        shared_changed_functions = sorted(list(ours_names & theirs_names))

        ast_diff_summary = {
            "ours": {fc.name: fc.change_type for fc in ours_changed_fns},
            "theirs": {fc.name: fc.change_type for fc in theirs_changed_fns},
            "shared": shared_changed_functions,
        }

        return MergeContext(
            file_path=file_path,
            base=base_version,
            ours=ours_context,
            theirs=theirs_context,
            conflict_markers=conflict_markers,
            ast_diff_summary=ast_diff_summary,
            shared_changed_functions=shared_changed_functions,
        )

    def ingest_all_conflicts(
        self,
        branch_a: str,
        branch_b: str,
    ) -> list[MergeContext]:
        """
        Ingest context for ALL conflicting Python files in the merge.

        Args:
            branch_a: Name or SHA of "ours" branch.
            branch_b: Name or SHA of "theirs" branch.

        Returns:
            list of MergeContext objects for all conflicting Python files.
        """
        conflicting_files = git_ops.get_conflicting_files(self.repo_path)
        contexts: list[MergeContext] = []

        for fpath in conflicting_files:
            if fpath.endswith(".py"):
                contexts.append(self.ingest(branch_a, branch_b, fpath))

        return contexts

    def _compute_merge_base(self, branch_a: str, branch_b: str) -> str:
        """Run git merge-base and return the commit SHA."""
        return git_ops.get_merge_base(self.repo_path, branch_a, branch_b)

    def _extract_file_version(self, commit: str, file_path: str) -> FileVersion:
        """Extract file content and metadata at a specific commit."""
        content = git_ops.get_file_at_commit(self.repo_path, commit, file_path)
        meta = git_ops.get_commit_metadata(self.repo_path, commit, file_path)
        return FileVersion(
            content=content,
            commit_hash=meta["commit_hash"],
            commit_message=meta["commit_message"],
            author=meta["author"],
            timestamp=meta["timestamp"],
        )

    def _analyze_ast_changes(
        self, base_src: str, branch_src: str
    ) -> list[FunctionChange]:
        """Compare ASTs and return function-level change descriptors."""
        base_fns = extract_functions(base_src)
        branch_fns = extract_functions(branch_src)

        changes: list[FunctionChange] = []
        all_names = sorted(list(set(base_fns.keys()) | set(branch_fns.keys())))

        for name in all_names:
            change = analyze_function_change(
                name=name,
                base_node=base_fns.get(name),
                branch_node=branch_fns.get(name),
                base_src=base_src,
                branch_src=branch_src,
            )
            if change is not None:
                changes.append(change)

        return changes

    def analyze_blast_radius(
        self, base_src: str, ours_src: str, theirs_src: str
    ) -> BlastRadius:
        """
        Compute AST-guided blast radius isolation using the BlastRadiusAnalyzer (Section 7.1).
        """
        return self.blast_analyzer.analyze(base_src, ours_src, theirs_src)
