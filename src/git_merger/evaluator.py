"""Differential Evaluator comparing parallel execution results (Section 3.2.5)."""

from typing import Any, Optional

from git_merger.models import (
    EvaluationVerdict,
    MergeContext,
    ParallelExecutionResult,
)
from git_merger.pytest_parser import ParsedPytestOutput, parse_pytest_output


class DifferentialEvaluator:
    """
    Evaluates parallel execution results from base, ours, theirs, and candidate sandboxes.
    Verifies that candidate M* preserves behavioral invariants of both branches
    without introducing functional regressions.
    """

    def evaluate(
        self,
        results: ParallelExecutionResult,
        context: Optional[MergeContext] = None,
    ) -> EvaluationVerdict:
        """
        Analyze parallel execution outputs and return an EvaluationVerdict.

        Args:
            results: ParallelExecutionResult with base, ours, theirs, and candidate outputs.
            context: Optional MergeContext for AST and change metadata.

        Returns:
            EvaluationVerdict indicating PASS/FAIL with detailed failure diagnostics.
        """
        # 1. Parse outputs from all four sandbox runs
        base_parsed = parse_pytest_output(results.base_result.stdout)
        ours_parsed = parse_pytest_output(results.ours_result.stdout)
        theirs_parsed = parse_pytest_output(results.theirs_result.stdout)
        cand_parsed = parse_pytest_output(results.candidate_result.stdout)

        # Build outcome maps: test_name -> outcome ("passed" | "failed" | "error")
        base_outcomes = {d.name: d.outcome for d in base_parsed.test_details}
        ours_outcomes = {d.name: d.outcome for d in ours_parsed.test_details}
        theirs_outcomes = {d.name: d.outcome for d in theirs_parsed.test_details}
        cand_outcomes = {d.name: d.outcome for d in cand_parsed.test_details}
        cand_details = {d.name: d for d in cand_parsed.test_details}

        all_test_names = set(cand_outcomes.keys()) | set(ours_outcomes.keys()) | set(theirs_outcomes.keys())

        failures: list[dict[str, Any]] = []
        lost_branch_a: list[str] = []
        lost_branch_b: list[str] = []
        regressions: list[str] = []

        # 2. Evaluate candidate against each test
        for test_name in sorted(all_test_names):
            cand_outcome = cand_outcomes.get(test_name, "missing")
            b_out = base_outcomes.get(test_name, "unknown")
            a_out = ours_outcomes.get(test_name, "unknown")
            t_out = theirs_outcomes.get(test_name, "unknown")

            if cand_outcome == "passed":
                continue

            # Candidate failed or errored on this test
            detail = cand_details.get(test_name)
            msg = detail.message if detail else "Test did not pass"
            tb = detail.traceback if detail else results.candidate_result.stderr

            branch_source = "unknown"

            # Case A: Passed in Branch A (ours), but failed in candidate
            if a_out == "passed" and t_out != "passed":
                branch_source = "ours"
                lost_branch_a.append(test_name)
                expected = "passed (as established by Branch A)"
            # Case B: Passed in Branch B (theirs), but failed in candidate
            elif t_out == "passed" and a_out != "passed":
                branch_source = "theirs"
                lost_branch_b.append(test_name)
                expected = "passed (as established by Branch B)"
            # Case C: Passed in both parents / base, but broke in candidate (regression)
            elif a_out == "passed" and t_out == "passed":
                branch_source = "both"
                regressions.append(test_name)
                expected = "passed (shared behavior in both branches)"
            elif b_out == "passed":
                branch_source = "base"
                regressions.append(test_name)
                expected = "passed (baseline behavior preserved in both branches)"
            else:
                expected = "passed"

            failures.append({
                "test_name": test_name,
                "expected": expected,
                "actual": cand_outcome,
                "branch_source": branch_source,
                "message": msg,
                "traceback": tb,
            })

        # Check syntax / runtime error in candidate execution
        if results.candidate_result.exit_code != 0 and not failures:
            failures.append({
                "test_name": "execution",
                "expected": "exit code 0",
                "actual": f"exit code {results.candidate_result.exit_code}",
                "branch_source": "candidate",
                "message": results.candidate_result.stderr.strip() or "Candidate execution failed",
                "traceback": results.candidate_result.stderr,
            })

        branch_a_preserved = len(lost_branch_a) == 0
        branch_b_preserved = len(lost_branch_b) == 0
        regression_detected = len(regressions) > 0 or (results.candidate_result.exit_code != 0 and cand_parsed.total == 0)

        passed = (
            results.candidate_result.exit_code == 0
            and len(failures) == 0
            and branch_a_preserved
            and branch_b_preserved
            and not regression_detected
        )

        # 3. Format human-readable summary
        if passed:
            summary = (
                f"100% Differential Verification Parity achieved. "
                f"All {cand_parsed.passed} invariant tests passed across Branch A and Branch B."
            )
        else:
            reasons = []
            if not branch_a_preserved:
                reasons.append(f"dropped {len(lost_branch_a)} invariant(s) from Branch A ({', '.join(lost_branch_a)})")
            if not branch_b_preserved:
                reasons.append(f"dropped {len(lost_branch_b)} invariant(s) from Branch B ({', '.join(lost_branch_b)})")
            if regression_detected:
                reasons.append(f"introduced {len(regressions)} functional regression(s)")
            if not reasons:
                reasons.append(f"{len(failures)} test failure(s)")
            summary = f"Differential Verification FAILED: Candidate M* {'; '.join(reasons)}."

        return EvaluationVerdict(
            passed=passed,
            failures=failures,
            summary=summary,
            branch_a_preserved=branch_a_preserved,
            branch_b_preserved=branch_b_preserved,
            regression_detected=regression_detected,
            raw_outputs=results,
        )
