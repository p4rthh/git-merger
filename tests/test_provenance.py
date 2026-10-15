"""Unit tests for Phase 5 ProvenanceReporter and models."""

from __future__ import annotations

from git_merger.models import EvaluationVerdict, RepairAttempt, ResolutionReport
from git_merger.provenance import (
    FunctionProvenance,
    ProvenanceLine,
    ProvenanceReporter,
    ResolutionProvenance,
)


def test_provenance_line_and_function_instantiation():
    """Verify ProvenanceLine and FunctionProvenance fields and defaults."""
    line = ProvenanceLine(
        line_number=42,
        content="return x * 2",
        source="ours",
        source_line=30,
        rationale="Adapted from Branch A implementation",
    )
    assert line.line_number == 42
    assert line.source == "ours"
    assert line.source_line == 30

    fn = FunctionProvenance(
        name="compute_metrics",
        primary_source="both",
        changes_from_ours=["Added caching parameter"],
        changes_from_theirs=["Added rate limiting"],
        rationale="Preserved both caching and rate limiting features",
    )
    assert fn.name == "compute_metrics"
    assert fn.primary_source == "both"
    assert len(fn.changes_from_ours) == 1
    assert len(fn.changes_from_theirs) == 1


def test_provenance_reporter_generate_markdown():
    """Verify ProvenanceReporter produces compliant PR markdown."""
    fn_prov = FunctionProvenance(
        name="process",
        primary_source="both",
        changes_from_ours=["Signature updated to add strict=False"],
        changes_from_theirs=["Added isinstance validation checks"],
        rationale="Unified signature with validation checks",
    )
    verdict = EvaluationVerdict(
        passed=True,
        failures=[],
        summary="All differential tests passed.",
        branch_a_preserved=True,
        branch_b_preserved=True,
        regression_detected=False,
    )

    attempt = RepairAttempt(
        iteration=1,
        previous_candidate="def process(): pass",
        failure_report=verdict,
        new_candidate="def process(data, strict=False): pass",
        new_explanation="Fixed signature mismatch and adapted nil-checks",
    )

    prov = ResolutionProvenance(
        file_path="processor.py",
        functions=[fn_prov],
        conflict_type="signature_body",
        overall_strategy="AST-guided semantic synthesis",
        confidence=0.95,
        token_cost={"total_tokens": 12500},
        repair_history=[attempt],
        test_summary=verdict.summary,
    )

    reporter = ProvenanceReporter()
    markdown = reporter.generate_markdown(prov)

    # Check key Markdown sections specified in Section 7.2
    assert "## 🔀 Semantic Merge Resolution Report" in markdown
    assert "**File:** `processor.py`" in markdown
    assert "**Conflict Type:** `signature_body`" in markdown
    assert "**Confidence:** 95%" in markdown
    assert "### Per-Function Provenance" in markdown
    assert "| `process` | `both` |" in markdown
    assert "### Differential Verification" in markdown
    assert "✅ **Verdict:** All differential tests passed." in markdown
    assert "### Repair History" in markdown
    assert "Fixed signature mismatch and adapted nil-checks" in markdown
    assert "### Resource Utilization" in markdown
    assert "12,500" in markdown


def test_provenance_reporter_from_resolution_report():
    """Verify ProvenanceReporter works directly on ResolutionReport."""
    report = ResolutionReport(
        status="success",
        final_candidate="def test(): pass",
        commit_sha="abcdef123",
        provenance={
            "process": {
                "source": "both",
                "rationale": "Harmonized signatures",
                "changes_applied": ["Branch A signature", "Branch B checks"],
            }
        },
        token_usage={"total_tokens": 4000},
    )

    reporter = ProvenanceReporter()
    markdown = reporter.generate_markdown(report)

    assert "## 🔀 Semantic Merge Resolution Report" in markdown
    assert "`process`" in markdown
    assert "Harmonized signatures" in markdown
