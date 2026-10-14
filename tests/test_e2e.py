
"""End-to-end orchestration tests for Scenario A.

Run with:
    pytest -v tests/test_e2e.py
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from git_merger.models import EvaluationVerdict
from git_merger.state import AgentState, AgentStateMachine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_A = PROJECT_ROOT / "tests" / "fixtures" / "scenario_a"

EXPECTED_STATES = [
    "IDLE",
    "INGESTING",
    "SYNTHESIZING",
    "GENERATING_HARNESS",
    "SANDBOX_EVALUATING",
    "SUCCESS",
    "COMMITTED",
]


def load_scenario_a():
    """Load the Scenario A benchmark files."""
    possible_dirs = [
        SCENARIO_A,
        PROJECT_ROOT / "tests" / "fixtures" / "scenario_a",
        PROJECT_ROOT / "tests" / "fixtures" / "scenario_a",
    ]

    fixture_dir = next(
        (path for path in possible_dirs if path.is_dir()),
        None,
    )

    if fixture_dir is None:
        pytest.fail(
            "Scenario A fixtures were not found. Expected files under "
            "tests/fixtures/scenario_a/: base.py, ours.py, "
            "theirs.py, and expected_merge.py."
        )

    required = ["base.py", "ours.py", "theirs.py", "expected_merge.py"]
    missing = [name for name in required if not (fixture_dir / name).exists()]

    if missing:
        pytest.fail(f"Missing Scenario A fixture files: {missing}")

    return {
        name: (fixture_dir / name).read_text(encoding="utf-8")
        for name in required
    }


@pytest.fixture
def scenario_a():
    return load_scenario_a()


@pytest.fixture
def mock_context(scenario_a):
    """Build a lightweight context carrying the Scenario A versions."""
    return SimpleNamespace(
        file_path="process.py",
        base=SimpleNamespace(content=scenario_a["base.py"]),
        ours=SimpleNamespace(
            file_version=SimpleNamespace(content=scenario_a["ours.py"]),
            changed_functions=[],
            commit_log=[],
            diff_from_base="",
            full_diff_patch="",
        ),
        theirs=SimpleNamespace(
            file_version=SimpleNamespace(content=scenario_a["theirs.py"]),
            changed_functions=[],
            commit_log=[],
            diff_from_base="",
            full_diff_patch="",
        ),
        conflict_markers=None,
        ast_diff_summary={},
        shared_changed_functions=["process", "summarize"],
    )


def test_scenario_a_fixture_contains_both_branch_changes(scenario_a):
    """The benchmark must include the signature change and safety checks."""
    ours = scenario_a["ours.py"]
    theirs = scenario_a["theirs.py"]
    expected = scenario_a["expected_merge.py"]

    assert "strict=False" in ours
    assert "data is None" in theirs
    assert "Expected list" in theirs

    assert "strict=False" in expected
    assert "data is None" in expected
    assert "Expected list" in expected
    assert "if not processed" in expected


def test_state_machine_expected_success_path():
    """Validate the expected state path for a successful Scenario A run."""
    machine = AgentStateMachine()

    machine.transition("conflict_detected")
    machine.transition("context_ready")
    machine.transition("candidate_produced")
    machine.transition("harness_ready")
    machine.transition("all_tests_passed")
    machine.transition("commit_created")

    actual_states = ["IDLE"] + [
        entry["to_state"] for entry in machine.history()
    ]

    assert actual_states == EXPECTED_STATES
    assert machine.current_state == AgentState.COMMITTED

    for entry in machine.history():
        assert entry["valid"] is True
        assert entry["timestamp"]


@pytest.mark.asyncio
async def test_orchestrator_wires_pipeline_and_executes_four_sandboxes(
    mock_context,
    scenario_a,
):
    """Exercise the orchestration path without calling a real model or API."""
    from git_merger import orchestrator

    candidate_code = scenario_a["expected_merge.py"]

    synthesis = SimpleNamespace(
        candidate_code=candidate_code,
        per_function_provenance={
            "process": "combined Branch A and Branch B",
            "summarize": "preserved empty-input guard",
        },
        token_usage={"prompt_tokens": 100, "completion_tokens": 80},
    )

    harness = SimpleNamespace(
        test_code="def test_scenario_a():\n    assert True\n",
        conftest_code="",
        target_functions=["process", "summarize"],
        test_count=1,
        pip_requirements=[],
        hypothesis_strategies=[],
    )

    passing_results = SimpleNamespace(
        base_result=SimpleNamespace(stdout="1 passed", exit_code=0),
        ours_result=SimpleNamespace(stdout="1 passed", exit_code=0),
        theirs_result=SimpleNamespace(stdout="1 passed", exit_code=0),
        candidate_result=SimpleNamespace(stdout="1 passed", exit_code=0),
        all_passed=True,
    )

    passing_verdict = EvaluationVerdict(
        passed=True,
        failures=[],
        summary="All differential tests passed.",
        branch_a_preserved=True,
        branch_b_preserved=True,
        regression_detected=False,
        raw_outputs=passing_results,
    )

    # Mock model-facing components so the test never makes API requests.
    fake_synthesizer = MagicMock()
    fake_synthesizer.synthesize.return_value = synthesis

    fake_harness_generator = MagicMock()
    fake_harness_generator.generate.return_value = harness
    fake_harness_generator.last_token_usage = {
        "prompt_tokens": 40,
        "completion_tokens": 30,
    }

    fake_provider = MagicMock()
    fake_provider.execute = AsyncMock()

    # Simulate four sandbox executions and return one aggregate result.
    fake_sandbox_manager = MagicMock()
    fake_sandbox_manager.execute_differential = AsyncMock(
        return_value=passing_results
    )

    fake_evaluator = MagicMock()
    fake_evaluator.evaluate.return_value = passing_verdict

    fake_ingester = MagicMock()
    fake_ingester.ingest.return_value = mock_context

    # Prevent a real Git commit from being created during the test.
    with (
        patch.object(
            orchestrator,
            "ContextIngester",
            return_value=fake_ingester,
        ),
        patch.object(
            orchestrator,
            "SemanticSynthesizer",
            return_value=fake_synthesizer,
        ),
        patch.object(
            orchestrator,
            "HarnessGenerator",
            return_value=fake_harness_generator,
        ),
        patch.object(
            orchestrator,
            "SandboxManager",
            return_value=fake_sandbox_manager,
        ),
        patch.object(
            orchestrator,
            "DifferentialEvaluator",
            return_value=fake_evaluator,
        ),
        patch.object(
            orchestrator,
            "_commit_verified_candidate",
            return_value="test-commit-sha",
            create=True,
        ),
    ):
        # Adjust these arguments if your orchestrator's signature differs.
        report = await orchestrator.resolve_merge_conflict(
            repo_path=str(PROJECT_ROOT),
            branch_a="ours",
            branch_b="theirs",
            file_path="process.py",
            max_repair_iterations=1,
            model_client=MagicMock(),
            sandbox_provider=fake_provider,
        )

    # (1) Ingestion received the correct branches and target file.
    fake_ingester.ingest.assert_called_once_with(
        "ours", "theirs", "process.py"
    )

    # (2) Synthesis received the ingested MergeContext.
    fake_synthesizer.synthesize.assert_called_once_with(mock_context)

    # (3) The harness generator received the context and candidate.
    fake_harness_generator.generate.assert_called_once_with(
        mock_context,
        candidate_code,
    )

    # (4) The differential runner was called once for all four sandboxes.
    fake_sandbox_manager.execute_differential.assert_awaited_once_with(
        mock_context,
        candidate_code,
        harness,
    )

    # (5) Evaluation returns a passing verdict.
    fake_evaluator.evaluate.assert_called_once_with(
        passing_results,
        mock_context,
    )
    assert passing_verdict.passed is True

    # The result should represent a successful resolution.
    assert report.status == "success"
    assert report.final_candidate == candidate_code
    assert report.commit_sha == "test-commit-sha"
