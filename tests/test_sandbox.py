"""Integration tests for Nebius Sandbox Provider, SandboxManager, and DifferentialEvaluator (Phase 3)."""

import asyncio
import os
import pytest

from git_merger.evaluator import DifferentialEvaluator
from git_merger.models import (
    BranchContext,
    ExecutionResult,
    FileVersion,
    MergeContext,
    ParallelExecutionResult,
    SandboxSpec,
    TestHarness,
)
from git_merger.pytest_parser import parse_pytest_output
from git_merger.sandbox import MockSandboxProvider, get_sandbox_provider
from git_merger.sandbox_manager import SandboxManager


# =========================================================================
# 1. Sandbox Provider Abstraction Tests
# =========================================================================

@pytest.mark.asyncio
async def test_mock_sandbox_lifecycle():
    """Test full lifecycle: create, upload, execute, snapshot, branch, destroy."""
    provider = MockSandboxProvider()
    spec = SandboxSpec(timeout_seconds=5)

    sandbox_id = await provider.create(spec)
    assert sandbox_id.startswith("mock-sbx-")

    # Upload test script
    await provider.upload_files(
        sandbox_id,
        {
            "hello.py": "print('hello world')",
            "subdir/nested.txt": "nested content",
        },
    )

    # Execute
    res = await provider.execute(sandbox_id, "python hello.py")
    assert res.exit_code == 0
    assert "hello world" in res.stdout
    assert not res.timed_out

    # Snapshot and branch
    snap_id = await provider.snapshot(sandbox_id)
    assert snap_id.startswith("snap-")

    branch_id = await provider.branch_from(snap_id)
    assert branch_id.startswith("mock-branch-")

    # Verify branched sandbox has the uploaded files
    branch_res = await provider.execute(branch_id, "python hello.py")
    assert branch_res.exit_code == 0
    assert "hello world" in branch_res.stdout

    # Destroy
    await provider.destroy(sandbox_id)
    await provider.destroy(branch_id)


@pytest.mark.asyncio
async def test_mock_sandbox_timeout():
    """Test that command exceeding timeout is terminated with timed_out flag."""
    provider = MockSandboxProvider()
    spec = SandboxSpec(timeout_seconds=1)

    sandbox_id = await provider.create(spec)
    try:
        # Command that sleeps longer than timeout
        res = await provider.execute(
            sandbox_id,
            "python -c 'import time; time.sleep(5)'",
            timeout_seconds=1,
        )
        assert res.timed_out is True
        assert res.exit_code == 124
    finally:
        await provider.destroy(sandbox_id)


def test_get_sandbox_provider_factory():
    """Test factory returns appropriate provider."""
    mock_p = get_sandbox_provider("mock")
    assert isinstance(mock_p, MockSandboxProvider)


# =========================================================================
# 2. Pytest Output Parser Tests
# =========================================================================

def test_pytest_output_parser_verbose_passing():
    """Test parser on verbose pytest passing output."""
    sample_stdout = """
test_differential.py::test_process_normal PASSED
test_differential.py::test_process_strict PASSED
============================== 2 passed in 0.05s ==============================
"""
    parsed = parse_pytest_output(sample_stdout)
    assert parsed.passed == 2
    assert parsed.failed == 0
    assert parsed.all_passed is True
    assert len(parsed.test_details) == 2
    assert parsed.test_details[0].name == "test_process_normal"
    assert parsed.test_details[0].outcome == "passed"


def test_pytest_output_parser_with_failures():
    """Test parser extracts failure messages and tracebacks."""
    sample_stdout = """
test_differential.py::test_process_normal PASSED
test_differential.py::test_process_nil FAILED

=================================== FAILURES ===================================
______________________________ test_process_nil _______________________________

    def test_process_nil():
>       assert process(None) == []
E       TypeError: Expected list, got NoneType

test_differential.py:15: TypeError
=========================== 1 failed, 1 passed in 0.12s ===========================
"""
    parsed = parse_pytest_output(sample_stdout)
    assert parsed.passed == 1
    assert parsed.failed == 1
    assert parsed.all_passed is False

    failed_test = next(d for d in parsed.test_details if d.name == "test_process_nil")
    assert failed_test.outcome == "failed"
    assert "TypeError: Expected list, got NoneType" in failed_test.message
    assert "assert process(None) == []" in failed_test.traceback


# =========================================================================
# 3. SandboxManager & Differential Parallel Execution Tests
# =========================================================================

SAMPLE_TEST_HARNESS_CODE = """
import os
import pytest
from target_module import process, summarize

def test_process_base_numbers():
    assert process([1, 2, 3]) == [2, 4, 6]

def test_process_strict_flag():
    # Branch A added strict parameter
    try:
        res = process([1, "invalid", 3], strict=False)
        assert res == [2, 6]
    except TypeError:
        # Base/Theirs don't accept strict arg
        pytest.fail("strict flag not supported")

def test_process_nil_safety():
    # Branch B added nil check
    try:
        assert process(None) == []
    except TypeError:
        pytest.fail("nil check missing")
"""


@pytest.fixture
def scenario_a_merge_context(scenario_a_sources):
    """Build a complete MergeContext fixture for Scenario A."""
    base_ver = FileVersion(
        content=scenario_a_sources["base"],
        commit_hash="c_base",
        commit_message="base",
        author="Dev",
        timestamp="2026-01-01",
    )
    ours_ver = FileVersion(
        content=scenario_a_sources["ours"],
        commit_hash="c_ours",
        commit_message="strict flag",
        author="Dev A",
        timestamp="2026-01-02",
    )
    theirs_ver = FileVersion(
        content=scenario_a_sources["theirs"],
        commit_hash="c_theirs",
        commit_message="nil check",
        author="Dev B",
        timestamp="2026-01-03",
    )

    ours_ctx = BranchContext(
        file_version=ours_ver,
        commit_log=["strict flag"],
        diff_from_base="",
        changed_functions=[],
        full_diff_patch="",
    )
    theirs_ctx = BranchContext(
        file_version=theirs_ver,
        commit_log=["nil check"],
        diff_from_base="",
        changed_functions=[],
        full_diff_patch="",
    )

    return MergeContext(
        file_path="process.py",
        base=base_ver,
        ours=ours_ctx,
        theirs=theirs_ctx,
        conflict_markers=None,
        ast_diff_summary={},
        shared_changed_functions=["process", "summarize"],
    )


@pytest.mark.asyncio
async def test_sandbox_manager_parallel_differential_execution(
    scenario_a_merge_context, scenario_a_sources
):
    """Test SandboxManager runs 4 sandboxes in parallel and collects outputs."""
    manager = SandboxManager(MockSandboxProvider())
    harness = TestHarness(
        test_code=SAMPLE_TEST_HARNESS_CODE,
        conftest_code="",
        requirements=[],
        target_functions=["process"],
        estimated_runtime_seconds=10,
        test_count=3,
    )

    # Candidate has the unified resolution
    candidate_code = scenario_a_sources["expected"]

    parallel_res = await manager.execute_differential(
        context=scenario_a_merge_context,
        candidate_code=candidate_code,
        harness=harness,
    )

    assert isinstance(parallel_res, ParallelExecutionResult)
    assert parallel_res.candidate_result.exit_code == 0
    # Candidate M* should pass all 3 tests
    assert "3 passed" in parallel_res.candidate_result.stdout

    # Base fails on strict flag and nil check
    assert parallel_res.base_result.exit_code != 0

    # Branch A passes strict flag but fails on nil check
    assert parallel_res.ours_result.exit_code != 0
    assert "test_process_strict_flag PASSED" in parallel_res.ours_result.stdout
    assert "test_process_nil_safety FAILED" in parallel_res.ours_result.stdout

    # Branch B passes nil check but fails on strict flag
    assert parallel_res.theirs_result.exit_code != 0
    assert "test_process_nil_safety PASSED" in parallel_res.theirs_result.stdout
    assert "test_process_strict_flag FAILED" in parallel_res.theirs_result.stdout


# =========================================================================
# 4. DifferentialEvaluator Verdict Tests
# =========================================================================

def test_differential_evaluator_100_percent_parity():
    """Test evaluator marks success when candidate passes both branch requirements."""
    evaluator = DifferentialEvaluator()

    mock_base = ExecutionResult(
        exit_code=1,
        stdout="""
test.py::test_common PASSED
test.py::test_strict FAILED
test.py::test_nil FAILED
""",
        stderr="",
        duration_ms=10,
        timed_out=False,
    )
    mock_ours = ExecutionResult(
        exit_code=1,
        stdout="""
test.py::test_common PASSED
test.py::test_strict PASSED
test.py::test_nil FAILED
""",
        stderr="",
        duration_ms=10,
        timed_out=False,
    )
    mock_theirs = ExecutionResult(
        exit_code=1,
        stdout="""
test.py::test_common PASSED
test.py::test_strict FAILED
test.py::test_nil PASSED
""",
        stderr="",
        duration_ms=10,
        timed_out=False,
    )
    mock_candidate = ExecutionResult(
        exit_code=0,
        stdout="""
test.py::test_common PASSED
test.py::test_strict PASSED
test.py::test_nil PASSED
============================== 3 passed in 0.05s ==============================
""",
        stderr="",
        duration_ms=10,
        timed_out=False,
    )

    parallel_results = ParallelExecutionResult(
        base_result=mock_base,
        ours_result=mock_ours,
        theirs_result=mock_theirs,
        candidate_result=mock_candidate,
        all_passed=False,
    )

    verdict = evaluator.evaluate(parallel_results)

    assert verdict.passed is True
    assert verdict.branch_a_preserved is True
    assert verdict.branch_b_preserved is True
    assert verdict.regression_detected is False
    assert len(verdict.failures) == 0
    assert "100% Differential Verification Parity" in verdict.summary


def test_differential_evaluator_dropped_branch_b_invariant():
    """Test evaluator catches when candidate drops Branch B's fix (e.g. nil check)."""
    evaluator = DifferentialEvaluator()

    mock_base = ExecutionResult(
        exit_code=1,
        stdout="test.py::test_nil FAILED\n",
        stderr="",
        duration_ms=10,
        timed_out=False,
    )
    mock_ours = ExecutionResult(
        exit_code=1,
        stdout="test.py::test_nil FAILED\n",
        stderr="",
        duration_ms=10,
        timed_out=False,
    )
    mock_theirs = ExecutionResult(
        exit_code=0,
        stdout="test.py::test_nil PASSED\n",
        stderr="",
        duration_ms=10,
        timed_out=False,
    )
    # Candidate dropped the nil check!
    mock_candidate = ExecutionResult(
        exit_code=1,
        stdout="""
test.py::test_nil FAILED
=================================== FAILURES ===================================
________________________________ test_nil ________________________________
    def test_nil():
>       assert process(None) == []
E       TypeError: 'NoneType' object is not iterable
=========================== 1 failed in 0.05s ===========================
""",
        stderr="",
        duration_ms=10,
        timed_out=False,
    )

    parallel_results = ParallelExecutionResult(
        base_result=mock_base,
        ours_result=mock_ours,
        theirs_result=mock_theirs,
        candidate_result=mock_candidate,
        all_passed=False,
    )

    verdict = evaluator.evaluate(parallel_results)

    assert verdict.passed is False
    assert verdict.branch_b_preserved is False
    assert len(verdict.failures) == 1
    assert verdict.failures[0]["branch_source"] == "theirs"
    assert "dropped 1 invariant(s) from Branch B" in verdict.summary


def test_differential_evaluator_regression_detected():
    """Test evaluator catches when candidate breaks existing shared/base behavior."""
    evaluator = DifferentialEvaluator()

    mock_base = ExecutionResult(
        exit_code=0,
        stdout="test.py::test_base_calc PASSED\n",
        stderr="",
        duration_ms=10,
        timed_out=False,
    )
    mock_ours = ExecutionResult(
        exit_code=0,
        stdout="test.py::test_base_calc PASSED\n",
        stderr="",
        duration_ms=10,
        timed_out=False,
    )
    mock_theirs = ExecutionResult(
        exit_code=0,
        stdout="test.py::test_base_calc PASSED\n",
        stderr="",
        duration_ms=10,
        timed_out=False,
    )
    mock_candidate = ExecutionResult(
        exit_code=1,
        stdout="""
test.py::test_base_calc FAILED
=================================== FAILURES ===================================
________________________________ test_base_calc ________________________________
E   AssertionError: assert 10 == 20
""",
        stderr="",
        duration_ms=10,
        timed_out=False,
    )

    parallel_results = ParallelExecutionResult(
        base_result=mock_base,
        ours_result=mock_ours,
        theirs_result=mock_theirs,
        candidate_result=mock_candidate,
        all_passed=False,
    )

    verdict = evaluator.evaluate(parallel_results)

    assert verdict.passed is False
    assert verdict.regression_detected is True
    assert "introduced 1 functional regression(s)" in verdict.summary
