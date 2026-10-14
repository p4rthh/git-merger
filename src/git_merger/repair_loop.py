
"""Closed-loop repair engine for semantic Git merge conflicts."""

import ast
from dataclasses import dataclass
from typing import Any

from git_merger.models import MergeContext, EvaluationVerdict


@dataclass
class RepairAttempt:
    """Record of a single repair iteration."""

    iteration: int
    previous_candidate: str
    failure_report: EvaluationVerdict
    new_candidate: str
    new_explanation: str
    token_usage: dict


@dataclass
class TokenBudget:
    """Track and enforce the total token usage limit."""

    max_total_tokens: int = 500_000
    prompt_tokens_used: int = 0
    completion_tokens_used: int = 0

    @property
    def total_used(self) -> int:
        return self.prompt_tokens_used + self.completion_tokens_used

    def has_remaining(self) -> bool:
        return self.total_used < self.max_total_tokens

    def record(self, usage: dict | None) -> None:
        """Add prompt and completion tokens from a model call."""
        if not usage:
            return

        self.prompt_tokens_used += int(usage.get("prompt_tokens", 0))
        self.completion_tokens_used += int(
            usage.get("completion_tokens", 0)
        )

    def summary(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens_used,
            "completion_tokens": self.completion_tokens_used,
            "total_tokens": self.total_used,
            "budget_remaining": max(
                0, self.max_total_tokens - self.total_used
            ),
            "utilization_pct": round(
                self.total_used / self.max_total_tokens * 100, 1
            ) if self.max_total_tokens > 0 else 0.0,
        }


class IterativeRepairLoop:
    """
    Repeatedly repair a merge candidate until it passes evaluation,
    reaches the iteration limit, or requires human intervention.
    """

    def __init__(
        self,
        synthesizer: Any,
        harness_generator: Any,
        sandbox_manager: Any,
        evaluator: Any,
        max_iterations: int = 5,
        token_budget: TokenBudget | None = None,
    ):
        if max_iterations < 0:
            raise ValueError("max_iterations cannot be negative.")

        self.synthesizer = synthesizer
        self.harness_generator = harness_generator
        self.sandbox_manager = sandbox_manager
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.token_budget = token_budget or TokenBudget()

        # Explains why the loop stopped or escalated.
        self.escalation_reason: str | None = None

    async def run(
        self,
        context: MergeContext,
        initial_candidate: str,
        initial_verdict: EvaluationVerdict,
    ) -> tuple[str, EvaluationVerdict, list[RepairAttempt]]:
        """
        Repair the candidate until tests pass or a stopping condition occurs.

        Returns:
            (final_candidate_code, final_verdict, repair_history)
        """
        candidate_code = initial_candidate
        verdict = initial_verdict
        repair_history: list[RepairAttempt] = []
        self.escalation_reason = None

        # Nothing to repair if the initial candidate already passes.
        if verdict.passed:
            return candidate_code, verdict, repair_history

        # Generate the initial differential test harness.
        harness = self.harness_generator.generate(
            context=context,
            candidate_code=candidate_code,
        )
        self.token_budget.record(
            getattr(self.harness_generator, "last_token_usage", {})
        )

        for iteration in range(1, self.max_iterations + 1):

            # Stop before another model call if the budget is exhausted.
            if not self.token_budget.has_remaining():
                self.escalation_reason = "Token budget exhausted."
                break

            previous_candidate = candidate_code
            previous_verdict = verdict

            # Ask the repair model to improve the failing candidate.
            repair_result = self.synthesizer.repair(
                context=context,
                previous_candidate=previous_candidate,
                verdict=previous_verdict,
                iteration=iteration,
                max_iterations=self.max_iterations,
                test_harness_code=self._get_harness_code(harness),
            )

            usage = repair_result.get("token_usage", {})
            self.token_budget.record(usage)

            # A genuine conflict cannot safely be resolved automatically.
            if repair_result.get("is_genuine_conflict", False):
                self.escalation_reason = repair_result.get(
                    "escalation_reason"
                ) or "The repair model detected a genuine conflict."
                break

            new_candidate = repair_result["candidate_code"]
            explanation = repair_result.get(
                "root_cause_analysis", ""
            )

            # Save the completed repair attempt.
            attempt = RepairAttempt(
                iteration=iteration,
                previous_candidate=previous_candidate,
                failure_report=previous_verdict,
                new_candidate=new_candidate,
                new_explanation=explanation,
                token_usage=usage,
            )
            repair_history.append(attempt)

            candidate_code = new_candidate

            # Regenerate the harness if the public function signatures
            # have changed, so tests reflect the updated interface.
            if self._signatures_changed(
                previous_candidate, candidate_code
            ):
                harness = self.harness_generator.generate(
                    context=context,
                    candidate_code=candidate_code,
                )
                self.token_budget.record(
                    getattr(
                        self.harness_generator,
                        "last_token_usage",
                        {},
                    )
                )

            # Execute all four versions and compare their behavior.
            execution_results = (
                await self.sandbox_manager.execute_differential(
                    context,
                    candidate_code,
                    harness,
                )
            )

            verdict = self.evaluator.evaluate(
                execution_results,
                context,
            )

            # Success: stop immediately.
            if verdict.passed:
                return candidate_code, verdict, repair_history

        # If we leave the loop without success, preserve the best candidate
        # and the latest verdict for reporting or human review.
        if self.escalation_reason is None:
            self.escalation_reason = (
                f"Failed after {self.max_iterations} repair iterations."
            )

        return candidate_code, verdict, repair_history

    @staticmethod
    def _get_harness_code(harness: Any) -> str:
        """Extract generated test code from a TestHarness object."""
        if isinstance(harness, str):
            return harness
        return getattr(harness, "test_code", "")

    @staticmethod
    def _function_signatures(source: str) -> dict[str, str]:
        """Extract function signatures for comparison."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Invalid candidate code warrants fresh harness generation.
            return {"__syntax_error__": source}

        signatures = {}

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                signature = ast.dump(
                    node.args, include_attributes=False
                )
                signature += "|returns="
                signature += ast.dump(
                    node.returns, include_attributes=False
                ) if node.returns else "None"

                signatures[node.name] = signature

        return signatures

    @classmethod
    def _signatures_changed(
        cls,
        previous_candidate: str,
        new_candidate: str,
    ) -> bool:
        """Return True when a function is added, removed, or changed."""
        return (
            cls._function_signatures(previous_candidate)
            != cls._function_signatures(new_candidate)
        )
