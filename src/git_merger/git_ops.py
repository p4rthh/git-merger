"""Git CLI operations wrapper for git-merger."""

import os
import subprocess
from typing import Optional

from git_merger.exceptions import GitError


def _run_git(
    repo_path: str,
    args: list[str],
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Execute a Git CLI command in the specified repository directory."""
    if not os.path.exists(repo_path):
        raise GitError(f"Repository path does not exist: {repo_path}")

    cmd = ["git", "-C", repo_path] + args
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as e:
        raise GitError(f"Git executable not found: {e}") from e
    except Exception as e:
        raise GitError(f"Failed to execute git command {cmd}: {e}") from e

    if check and result.returncode != 0:
        cmd_str = " ".join(cmd)
        err_msg = result.stderr.strip() or result.stdout.strip()
        raise GitError(
            f"Command '{cmd_str}' failed with exit code {result.returncode}: {err_msg}",
            returncode=result.returncode,
            stderr=result.stderr,
        )

    return result


def get_merge_base(repo_path: str, branch_a: str, branch_b: str) -> str:
    """
    Compute the lowest common ancestor (merge base) commit SHA between two branches/commits.

    Args:
        repo_path: Path to the Git repository.
        branch_a: Branch or commit SHA for "ours".
        branch_b: Branch or commit SHA for "theirs".

    Returns:
        The merge base commit SHA as a hex string.
    """
    result = _run_git(repo_path, ["merge-base", branch_a, branch_b], check=False)
    if result.returncode != 0 or not result.stdout.strip():
        err_msg = result.stderr.strip() or "No common ancestor found"
        raise GitError(
            f"Could not compute merge base between '{branch_a}' and '{branch_b}': {err_msg}",
            returncode=result.returncode,
            stderr=result.stderr,
        )
    return result.stdout.strip().splitlines()[0].strip()


def get_file_at_commit(repo_path: str, commit: str, file_path: str) -> str:
    """
    Extract file content at a specific commit or branch reference.

    Args:
        repo_path: Path to the Git repository.
        commit: Commit SHA, branch name, or tag.
        file_path: Relative path to the file in the repository.

    Returns:
        The content of the file at that revision.
    """
    spec = f"{commit}:{file_path}"
    result = _run_git(repo_path, ["show", spec], check=False)
    if result.returncode != 0:
        err_msg = result.stderr.strip() or "File or commit not found"
        raise GitError(
            f"Failed to get file '{file_path}' at commit '{commit}': {err_msg}",
            returncode=result.returncode,
            stderr=result.stderr,
        )
    return result.stdout


def get_commit_log(
    repo_path: str,
    from_commit: str,
    to_commit: str,
    file_path: Optional[str] = None,
) -> list[str]:
    """
    Retrieve one-line commit messages between two commits, optionally filtered by file path.

    Args:
        repo_path: Path to the Git repository.
        from_commit: Base commit SHA.
        to_commit: Target commit/branch SHA.
        file_path: Optional path to filter commits affecting this file.

    Returns:
        List of commit log strings (format: '<sha> <subject>').
    """
    args = ["log", f"{from_commit}..{to_commit}", "--oneline", "--no-merges"]
    if file_path:
        args += ["--", file_path]

    result = _run_git(repo_path, args, check=False)
    if result.returncode != 0:
        err_msg = result.stderr.strip() or "Failed to read commit log"
        raise GitError(
            f"Failed to get commit log between '{from_commit}' and '{to_commit}': {err_msg}",
            returncode=result.returncode,
            stderr=result.stderr,
        )

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines


def get_diff(
    repo_path: str,
    from_commit: str,
    to_commit: str,
    file_path: Optional[str] = None,
) -> str:
    """
    Retrieve unified diff between two commits, optionally filtered by file path.

    Args:
        repo_path: Path to the Git repository.
        from_commit: Base commit SHA.
        to_commit: Target commit/branch SHA.
        file_path: Optional path to restrict the diff.

    Returns:
        Unified diff string.
    """
    args = ["diff", from_commit, to_commit]
    if file_path:
        args += ["--", file_path]

    result = _run_git(repo_path, args, check=False)
    if result.returncode != 0:
        err_msg = result.stderr.strip() or "Failed to compute diff"
        raise GitError(
            f"Failed to get diff between '{from_commit}' and '{to_commit}': {err_msg}",
            returncode=result.returncode,
            stderr=result.stderr,
        )

    return result.stdout


def get_conflicting_files(repo_path: str) -> list[str]:
    """
    Retrieve list of unmerged / conflicting files in the repository working directory.

    Args:
        repo_path: Path to the Git repository.

    Returns:
        List of conflicting file paths relative to repo root.
    """
    # Check diff-filter=U first
    result = _run_git(repo_path, ["diff", "--name-only", "--diff-filter=U"], check=False)
    files: set[str] = set()
    if result.returncode == 0 and result.stdout.strip():
        for line in result.stdout.splitlines():
            line = line.strip()
            if line:
                files.add(line)

    # Also parse git status --porcelain for conflict indicators: UU, AA, UD, DU, etc.
    status_res = _run_git(repo_path, ["status", "--porcelain"], check=False)
    if status_res.returncode == 0 and status_res.stdout.strip():
        conflict_prefixes = {"UU", "AA", "UD", "DU", "DD", "AU", "UA"}
        for line in status_res.stdout.splitlines():
            line = line.strip()
            if len(line) >= 4:
                code = line[:2]
                if code in conflict_prefixes:
                    fpath = line[3:].strip()
                    # Handle quoted paths if any
                    if fpath.startswith('"') and fpath.endswith('"'):
                        fpath = fpath[1:-1]
                    files.add(fpath)

    return sorted(files)


def get_commit_metadata(
    repo_path: str,
    commit: str,
    file_path: Optional[str] = None,
) -> dict:
    """
    Extract commit metadata (hash, author, date, message) for a specific commit or file.

    Returns:
        dict with keys: 'commit_hash', 'author', 'timestamp', 'commit_message'.
    """
    # Format: %H (full hash)%x00%an <%ae>%x00%aI (ISO 8601)%x00%B (raw body)
    args = ["log", "-1", "--format=%H%x00%an <%ae>%x00%aI%x00%B", commit]
    if file_path:
        args += ["--", file_path]

    result = _run_git(repo_path, args, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        # Fallback to commit hash only if log with file_path is empty
        args_no_file = ["log", "-1", "--format=%H%x00%an <%ae>%x00%aI%x00%B", commit]
        result = _run_git(repo_path, args_no_file, check=False)

    if result.returncode != 0 or not result.stdout.strip():
        return {
            "commit_hash": commit,
            "author": "Unknown",
            "timestamp": "",
            "commit_message": "",
        }

    parts = result.stdout.split("\x00", 3)
    return {
        "commit_hash": parts[0].strip() if len(parts) > 0 else commit,
        "author": parts[1].strip() if len(parts) > 1 else "Unknown",
        "timestamp": parts[2].strip() if len(parts) > 2 else "",
        "commit_message": parts[3].strip() if len(parts) > 3 else "",
    }


def read_working_tree_file(repo_path: str, file_path: str) -> Optional[str]:
    """Read working tree file content if it exists on disk."""
    full_path = os.path.join(repo_path, file_path)
    if os.path.isfile(full_path):
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError:
            return None
    return None
