"""Unit tests for Phase 5 AdaptiveTestPruner and TokenBudgetController."""

from __future__ import annotations

from git_merger.models import EvaluationVerdict, TestHarness
from git_merger.pruner import (
    AdaptiveTestPruner,
    PruningDecision,
    TokenBudgetController,
)


def test_pruning_decision_iterable():
    """Verify PruningDecision behaves as expected and is iterable."""
    decision = PruningDecision(
        tests_to_run=["test_one", "test_two"],
        tests_pruned=["test_three"],
        pruning_rationale={"test_three": "pruned"},
        estimated_savings_pct=33.3,
    )
    assert len(decision) == 2
    assert list(decision) == ["test_one", "test_two"]
    assert decision.estimated_savings_pct == 33.3


def test_adaptive_test_pruner_iteration_zero_runs_all():
    """Verify iteration 0 runs all tests without pruning (baseline)."""
    harness = TestHarness(
        test_code="def test_a(): pass\ndef test_b(): pass\ndef test_c(): pass\n",
        conftest_code="",
        requirements=[],
        target_functions=["a", "b", "c"],
        estimated_runtime_seconds=5,
        test_count=3,
    )

    pruner = AdaptiveTestPruner()
    decision = pruner.prune(harness, iteration=0)

    assert len(decision.tests_to_run) == 3
    assert set(decision.tests_to_run) == {"test_a", "test_b", "test_c"}
    assert len(decision.tests_pruned) == 0
    assert decision.estimated_savings_pct == 0.0


def test_adaptive_test_pruner_iteration_one_prunes_passing():
    """Verify iteration 1+ prioritizes failed tests + sampled passing tests."""
    tests = [f"test_func_{i}" for i in range(10)]
    harness = TestHarness(
        test_code="\n".join(f"def {t}(): pass" for t in tests),
        conftest_code="",
        requirements=[],
        target_functions=[],
        estimated_runtime_seconds=10,
        test_count=10,
    )

    # Simulate previous verdict where test_func_1 and test_func_3 failed
    previous_verdict = EvaluationVerdict(
        passed=False,
        failures=[{"test_name": "test_func_1"}, {"test_name": "test_func_3"}],
        summary="2 tests failed",
        branch_a_preserved=True,
        branch_b_preserved=False,
        regression_detected=False,
    )

    pruner = AdaptiveTestPruner(sample_rate=0.25)
    decision = pruner.prune(harness, iteration=1, previous_verdict=previous_verdict)

    # Failed tests MUST always be in tests_to_run
    assert "test_func_1" in decision.tests_to_run
    assert "test_func_3" in decision.tests_to_run

    # Total tests to run should be significantly less than 10
    assert len(decision.tests_to_run) < 10
    assert len(decision.tests_pruned) > 0
    assert decision.estimated_savings_pct > 0.0


def test_token_budget_controller():
    """Verify TokenBudgetController tracks phase usage and limits."""
    controller = TokenBudgetController(total_budget=150_000)

    # Check synthesis budget limit is 100,000
    assert controller.can_spend("synthesis", 50_000) is True
    assert controller.can_spend("synthesis", 120_000) is False

    controller.record("synthesis", 60_000)
    assert controller.phase_usage["synthesis"] == 60_000
    assert controller.should_downgrade_model("synthesis") is False

    controller.record("synthesis", 20_000)  # total 80k/100k = 80%
    assert controller.should_downgrade_model("synthesis") is True

    # Check total budget limit
    assert controller.can_spend("repair_per_iter", 80_000) is False  # 80k + 80k > 150k
