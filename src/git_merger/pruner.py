"""Adaptive Test Pruning & Token Budgeting.

Prunes redundant differential tests before sandbox execution to minimize latency
and cloud compute costs. Enforces per-phase token budgets to avoid overrun.
"""

from __future__ import annotations

import ast
import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from git_merger.models import BlastRadius, EvaluationVerdict, TestHarness


@dataclass
class PruningDecision:
    """Decision on which tests to run."""

    tests_to_run: list[str]  # Test function names to execute
    tests_pruned: list[str]  # Test function names skipped
    pruning_rationale: dict[str, str] = field(default_factory=dict)  # {test_name: reason}
    estimated_savings_pct: float = 0.0

    def __iter__(self):
        """Allow iterating directly over tests_to_run."""
        return iter(self.tests_to_run)

    def __len__(self):
        return len(self.tests_to_run)


class AdaptiveTestPruner:
    """Prune redundant tests before sandbox dispatch."""

    def __init__(self, model_client: Any | None = None, sample_rate: float = 0.2):
        self.client = model_client
        self.model = "nvidia/nemotron-3-nano-30b-a3b"
        self.sample_rate = sample_rate

    @staticmethod
    def extract_test_names(harness_or_tests: TestHarness | list[str] | str) -> list[str]:
        """Extract individual test function names from a harness, code string, or list."""
        if isinstance(harness_or_tests, list):
            return list(harness_or_tests)

        code = harness_or_tests.test_code if isinstance(harness_or_tests, TestHarness) else harness_or_tests
        test_names = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    test_names.append(node.name)
        except Exception:
            # Fallback regex parsing if syntax isn't valid standalone AST
            test_names = re.findall(r"def\s+(test_\w+)\s*\(", code)

        return test_names or ["test_differential"]

    def prune(
        self,
        harness: TestHarness | list[str] | str,
        blast_radius: Optional[BlastRadius] = None,
        iteration: int = 0,
        previous_verdict: Optional[EvaluationVerdict] = None,
    ) -> PruningDecision:
        """
        Analyze test harness and prune redundant tests.

        Strategy:
        1. On iteration 0: Run all tests (full coverage baseline).
        2. On iteration 1+: Run only tests that FAILED in previous iteration
           plus a random sample (default 20%) of passing tests (regression guard).
        3. Can leverage BlastRadius to focus on affected functions.
        """
        all_tests = self.extract_test_names(harness)
        total_count = len(all_tests)

        if total_count == 0:
            return PruningDecision(
                tests_to_run=[],
                tests_pruned=[],
                pruning_rationale={},
                estimated_savings_pct=0.0,
            )

        # Baseline: run all tests on first iteration or when no previous verdict exists
        if iteration == 0 or previous_verdict is None:
            return PruningDecision(
                tests_to_run=list(all_tests),
                tests_pruned=[],
                pruning_rationale={t: "Baseline full execution (iteration 0)" for t in all_tests},
                estimated_savings_pct=0.0,
            )

        # Identify failed tests from previous verdict
        failed_test_names: set[str] = set()
        for failure in getattr(previous_verdict, "failures", []):
            if isinstance(failure, dict):
                test_name = failure.get("test_name") or failure.get("test")
                if test_name:
                    failed_test_names.add(str(test_name))
            elif isinstance(failure, str):
                failed_test_names.add(failure)

        # Also check raw output text if structured failures list wasn't populated
        if not failed_test_names and getattr(previous_verdict, "raw_outputs", None):
            raw = previous_verdict.raw_outputs
            for out in [getattr(raw, "candidate_result", None), getattr(raw, "base_result", None)]:
                if out and hasattr(out, "stdout"):
                    for test_name in all_tests:
                        if f"FAILED {test_name}" in out.stdout or f"{test_name} FAILED" in out.stdout:
                            failed_test_names.add(test_name)

        # If previous verdict didn't identify specific test names, fall back to running all tests
        if not failed_test_names and not previous_verdict.passed:
            return PruningDecision(
                tests_to_run=list(all_tests),
                tests_pruned=[],
                pruning_rationale={t: "Retrying all tests due to unspecified failure origin" for t in all_tests},
                estimated_savings_pct=0.0,
            )

        passing_tests = [t for t in all_tests if t not in failed_test_names]

        # Sample passing tests for regression guard
        k = max(1, int(round(len(passing_tests) * self.sample_rate))) if passing_tests else 0
        sampled_passing = random.sample(passing_tests, min(k, len(passing_tests))) if passing_tests else []
        sampled_set = set(sampled_passing)

        tests_to_run: list[str] = []
        tests_pruned: list[str] = []
        rationale: dict[str, str] = {}

        for test in all_tests:
            if test in failed_test_names:
                tests_to_run.append(test)
                rationale[test] = "Failed in previous iteration; must be verified"
            elif test in sampled_set:
                tests_to_run.append(test)
                rationale[test] = "Sampled passing test (regression guard)"
            else:
                tests_pruned.append(test)
                rationale[test] = "Passed in previous iteration; pruned for performance"

        savings = (len(tests_pruned) / total_count * 100.0) if total_count > 0 else 0.0

        return PruningDecision(
            tests_to_run=tests_to_run,
            tests_pruned=tests_pruned,
            pruning_rationale=rationale,
            estimated_savings_pct=round(savings, 1),
        )


class TokenBudgetController:
    """Enforce per-phase and total token budgets."""

    PHASE_BUDGETS = {
        "synthesis": 100_000,
        "harness_gen": 50_000,
        "repair_per_iter": 80_000,
        "log_parsing": 20_000,
    }

    def __init__(self, total_budget: int = 500_000):
        self.total_budget = total_budget
        self.phase_usage: dict[str, int] = {}

    def can_spend(self, phase: str, estimated_tokens: int) -> bool:
        """Check if spending estimated_tokens in this phase is within budget."""
        phase_limit = self.PHASE_BUDGETS.get(phase, 50_000)
        phase_used = self.phase_usage.get(phase, 0)
        total_used = sum(self.phase_usage.values())

        return (
            phase_used + estimated_tokens <= phase_limit
            and total_used + estimated_tokens <= self.total_budget
        )

    def record(self, phase: str, tokens_used: int) -> None:
        """Record token usage for a specific phase."""
        self.phase_usage[phase] = self.phase_usage.get(phase, 0) + tokens_used

    def should_downgrade_model(self, phase: str) -> bool:
        """
        If budget is running low, recommend downgrading to a smaller model.
        Returns True if > 70% of phase budget is consumed.
        """
        phase_limit = self.PHASE_BUDGETS.get(phase, 50_000)
        phase_used = self.phase_usage.get(phase, 0)
        return (phase_used / phase_limit) > 0.7
