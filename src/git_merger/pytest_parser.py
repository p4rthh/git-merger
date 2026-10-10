"""Parser for pytest standard and verbose console output."""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TestResultDetail:
    """Individual test outcome and failure trace."""
    node_id: str
    name: str
    outcome: str  # "passed" | "failed" | "error" | "skipped"
    message: Optional[str] = None
    traceback: Optional[str] = None


@dataclass
class ParsedPytestOutput:
    """Structured summary of a pytest execution."""
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration_seconds: float = 0.0
    test_details: list[TestResultDetail] = field(default_factory=list)
    raw_stdout: str = ""

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors + self.skipped

    @property
    def all_passed(self) -> bool:
        return self.failed == 0 and self.errors == 0 and self.passed > 0


def parse_pytest_output(stdout: str) -> ParsedPytestOutput:
    """
    Parse stdout from pytest -v --tb=long runs into structured details.

    Handles lines like:
        test_differential.py::test_process_strict PASSED
        test_differential.py::test_process_nil FAILED
    and failure sections (FAILURES / ERRORS).
    """
    parsed = ParsedPytestOutput(raw_stdout=stdout)
    if not stdout or not stdout.strip():
        return parsed

    # 1. Parse individual test outcome lines
    # Matches: test_file.py::test_name PASSED / FAILED / ERROR / SKIPPED
    test_line_pattern = re.compile(
        r"^([^\s:]+\.py(?:::[\w<>\[\]\-\.]+)+)\s+(PASSED|FAILED|ERROR|SKIPPED)",
        re.MULTILINE,
    )

    details_by_id: dict[str, TestResultDetail] = {}

    for match in test_line_pattern.finditer(stdout):
        node_id = match.group(1).strip()
        outcome = match.group(2).lower()
        test_name = node_id.split("::")[-1]

        detail = TestResultDetail(node_id=node_id, name=test_name, outcome=outcome)
        details_by_id[node_id] = detail

        if outcome == "passed":
            parsed.passed += 1
        elif outcome == "failed":
            parsed.failed += 1
        elif outcome == "error":
            parsed.errors += 1
        elif outcome == "skipped":
            parsed.skipped += 1

    # 2. Extract failure / error traceback sections
    # Pytest failure sections are delimited by:
    # _________________________ test_name _________________________
    failure_block_pattern = re.compile(
        r"_{3,}\s+(.*?)\s+_{3,}\n(.*?)(?=\n_{3,}|\n={3,}|\Z)",
        re.DOTALL,
    )

    for match in failure_block_pattern.finditer(stdout):
        failed_test_name = match.group(1).strip()
        body = match.group(2).strip()

        # Find matching detail by name or substring
        for node_id, detail in details_by_id.items():
            if detail.name == failed_test_name or failed_test_name in node_id:
                detail.traceback = body
                # Extract the last error line if available (e.g. "E   AssertionError: ...")
                error_lines = [
                    line.strip() for line in body.splitlines() if line.startswith("E   ")
                ]
                if error_lines:
                    detail.message = error_lines[-1][4:].strip()
                elif "AssertionError" in body:
                    detail.message = "AssertionError"
                else:
                    detail.message = body.splitlines()[-1] if body.splitlines() else None
                break

    # 3. Fallback summary line matching: "X passed, Y failed in Zs"
    summary_pattern = re.search(
        r"(=+)\s*(.*?)\s*in\s+([0-9\.]+s?)\s*(=+)", stdout
    )
    if summary_pattern:
        summary_text = summary_pattern.group(2)
        duration_str = summary_pattern.group(3).rstrip("s")
        try:
            parsed.duration_seconds = float(duration_str)
        except ValueError:
            parsed.duration_seconds = 0.0

        # If individual counts were 0 (e.g. non-verbose run), parse numbers from summary
        if parsed.total == 0:
            passed_m = re.search(r"(\d+)\s+passed", summary_text)
            failed_m = re.search(r"(\d+)\s+failed", summary_text)
            errors_m = re.search(r"(\d+)\s+error", summary_text)
            skipped_m = re.search(r"(\d+)\s+skipped", summary_text)

            parsed.passed = int(passed_m.group(1)) if passed_m else 0
            parsed.failed = int(failed_m.group(1)) if failed_m else 0
            parsed.errors = int(errors_m.group(1)) if errors_m else 0
            parsed.skipped = int(skipped_m.group(1)) if skipped_m else 0

    parsed.test_details = list(details_by_id.values())
    return parsed
