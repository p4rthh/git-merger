"""Pytest fixtures for git-merger tests."""

import os
import subprocess
import pytest

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def scenario_a_sources():
    """Load the raw source code strings for Scenario A."""
    scenario_dir = os.path.join(FIXTURES_DIR, "scenario_a")
    with open(os.path.join(scenario_dir, "base.py"), "r", encoding="utf-8") as f:
        base_src = f.read()
    with open(os.path.join(scenario_dir, "ours.py"), "r", encoding="utf-8") as f:
        ours_src = f.read()
    with open(os.path.join(scenario_dir, "theirs.py"), "r", encoding="utf-8") as f:
        theirs_src = f.read()
    with open(os.path.join(scenario_dir, "expected_merge.py"), "r", encoding="utf-8") as f:
        expected_src = f.read()

    return {
        "base": base_src,
        "ours": ours_src,
        "theirs": theirs_src,
        "expected": expected_src,
    }


@pytest.fixture
def git_test_repo(tmp_path, scenario_a_sources):
    """
    Create a temporary Git repository configured with Scenario A history:
    - Base commit on 'main' containing process.py
    - Branch 'branch_a' modifying process.py (signature change)
    - Branch 'branch_b' modifying process.py (nil check bugfix)
    """
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()

    def run_git(*args):
        res = subprocess.run(
            ["git", "-C", str(repo_dir)] + list(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout.strip()

    run_git("init", "-b", "main")
    run_git("config", "user.name", "Test Committer")
    run_git("config", "user.email", "test@example.com")

    # 1. Base commit
    file_path = repo_dir / "process.py"
    file_path.write_text(scenario_a_sources["base"], encoding="utf-8")
    run_git("add", "process.py")
    run_git("commit", "-m", "Initial commit: basic process and summarize")
    base_sha = run_git("rev-parse", "HEAD")

    # 2. Branch A (ours)
    run_git("checkout", "-b", "branch_a")
    file_path.write_text(scenario_a_sources["ours"], encoding="utf-8")
    run_git("add", "process.py")
    run_git("commit", "-m", "Refactor: add strict flag to process and summarize")
    branch_a_sha = run_git("rev-parse", "HEAD")

    # 3. Branch B (theirs)
    run_git("checkout", "main")
    run_git("checkout", "-b", "branch_b")
    file_path.write_text(scenario_a_sources["theirs"], encoding="utf-8")
    run_git("add", "process.py")
    run_git("commit", "-m", "Bugfix: nil-check and div-by-zero protection in process")
    branch_b_sha = run_git("rev-parse", "HEAD")

    return {
        "repo_path": str(repo_dir),
        "file_path": "process.py",
        "branch_a": "branch_a",
        "branch_b": "branch_b",
        "base_sha": base_sha,
        "branch_a_sha": branch_a_sha,
        "branch_b_sha": branch_b_sha,
    }
