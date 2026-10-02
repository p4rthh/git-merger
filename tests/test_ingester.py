"""Tests for AST analysis, blast radius calculation, git ops, and context ingestion (Phase 1)."""

import ast
import subprocess
import pytest

from git_merger.ast_analysis import (
    BlastRadiusAnalyzer,
    build_call_graph,
    compute_transitive_callers,
    diff_functions,
    extract_functions,
    get_function_source,
    get_signature,
)
from git_merger.exceptions import ASTAnalysisError, GitError, IngestionError
from git_merger.git_ops import (
    get_commit_log,
    get_diff,
    get_file_at_commit,
    get_merge_base,
)
from git_merger.ingester import ContextIngester


# =========================================================================
# 1. AST Function Extraction Tests
# =========================================================================

def test_ast_function_extraction_scenario_a(scenario_a_sources):
    """Test AST function extraction correctly extracts all functions from Scenario A."""
    base_fns = extract_functions(scenario_a_sources["base"])
    assert "process" in base_fns
    assert "summarize" in base_fns
    assert len(base_fns) == 2

    ours_fns = extract_functions(scenario_a_sources["ours"])
    assert "process" in ours_fns
    assert "summarize" in ours_fns
    assert len(ours_fns) == 2

    theirs_fns = extract_functions(scenario_a_sources["theirs"])
    assert "process" in theirs_fns
    assert "summarize" in theirs_fns
    assert len(theirs_fns) == 2


def test_ast_function_source_extraction(scenario_a_sources):
    """Test get_function_source accurately slices the function source segment."""
    base_fns = extract_functions(scenario_a_sources["base"])
    process_src = get_function_source(scenario_a_sources["base"], base_fns["process"])
    assert "def process(data):" in process_src
    assert "return result" in process_src


def test_ast_extraction_empty_and_syntax_error():
    """Test empty source returns empty dict and invalid syntax raises ASTAnalysisError."""
    assert extract_functions("") == {}
    assert extract_functions("   \n") == {}

    with pytest.raises(ASTAnalysisError):
        extract_functions("def bad_syntax(:")


# =========================================================================
# 2. Signature Comparison & Diffing Tests
# =========================================================================

def test_signature_comparison_scenario_a(scenario_a_sources):
    """
    Test signature extraction and diffing on Scenario A:
    - Base -> Branch A: process signature changes (strict=False added), summarize signature changes.
    - Base -> Branch B: process signature unchanged (body changed), summarize signature unchanged.
    """
    base_fns = extract_functions(scenario_a_sources["base"])
    ours_fns = extract_functions(scenario_a_sources["ours"])
    theirs_fns = extract_functions(scenario_a_sources["theirs"])

    sig_base_process = get_signature(base_fns["process"])
    sig_ours_process = get_signature(ours_fns["process"])
    sig_theirs_process = get_signature(theirs_fns["process"])

    # Branch A added 'strict' argument and default
    assert "strict" not in sig_base_process["args"]
    assert "strict" in sig_ours_process["args"]
    assert sig_base_process != sig_ours_process

    # Branch B kept the exact same signature
    assert sig_base_process == sig_theirs_process

    # Test diff_functions
    diff_a = diff_functions(scenario_a_sources["base"], scenario_a_sources["ours"])
    assert diff_a["process"] == "modified"
    assert diff_a["summarize"] == "modified"

    diff_b = diff_functions(scenario_a_sources["base"], scenario_a_sources["theirs"])
    # Branch B modified body of process and summarize
    assert diff_b["process"] == "modified"
    assert diff_b["summarize"] == "modified"


def test_diff_functions_add_and_remove():
    """Test diff_functions detects added and removed functions."""
    base_code = "def existing_fn(): pass\ndef to_remove(): pass"
    branch_code = "def existing_fn(): pass\ndef newly_added(): pass"

    diff = diff_functions(base_code, branch_code)
    assert diff["newly_added"] == "added"
    assert diff["to_remove"] == "removed"
    assert "existing_fn" not in diff


# =========================================================================
# 3. Call Graph Construction Tests
# =========================================================================

def test_call_graph_construction_scenario_a(scenario_a_sources):
    """
    Test call graph construction on Scenario A:
    - summarize() calls process()
    - process() calls nothing within module
    """
    cg = build_call_graph(scenario_a_sources["base"])
    assert "summarize" in cg
    assert "process" in cg
    assert "process" in cg["summarize"]
    assert len(cg["process"]) == 0


def test_transitive_callers_calculation():
    """Test multi-hop caller graph traversal: A calls B, B calls C -> targets={C} -> affected={C, B, A}."""
    code = """
def c(): pass
def b(): c()
def a(): b()
def independent(): pass
"""
    cg = build_call_graph(code)
    assert "c" in cg["b"]
    assert "b" in cg["a"]

    transitive = compute_transitive_callers(cg, targets={"c"})
    assert transitive == {"a", "b", "c"}
    assert "independent" not in transitive


# =========================================================================
# 4. Blast Radius Computation Tests
# =========================================================================

def test_blast_radius_computation_scenario_a(scenario_a_sources):
    """
    Test BlastRadiusAnalyzer on Scenario A:
    Both process and summarize are directly modified in Branch A.
    Blast radius affected set must contain both functions.
    """
    analyzer = BlastRadiusAnalyzer()
    br = analyzer.analyze(
        base_src=scenario_a_sources["base"],
        ours_src=scenario_a_sources["ours"],
        theirs_src=scenario_a_sources["theirs"],
    )

    assert "process" in br.directly_changed
    assert "summarize" in br.directly_changed
    assert "process" in br.affected
    assert "summarize" in br.affected
    assert "summarize" in br.callers.get("process", set())
    assert len(br.unchanged) == 0


def test_blast_radius_with_unchanged_functions():
    """Test blast radius isolates unchanged functions that are not in the call path."""
    base = """
def helper(x): return x * 2
def core(x): return helper(x)
def unrelated(): return 42
"""
    # Ours modifies helper only
    ours = """
def helper(x): return x * 3
def core(x): return helper(x)
def unrelated(): return 42
"""
    theirs = base

    analyzer = BlastRadiusAnalyzer()
    br = analyzer.analyze(base, ours, theirs)

    assert br.directly_changed == {"helper"}
    # core calls helper, so core is affected transitively
    assert "helper" in br.affected
    assert "core" in br.affected
    # unrelated does not call helper and is unchanged
    assert "unrelated" in br.unchanged
    assert "unrelated" not in br.affected


# =========================================================================
# 5. ContextIngester End-to-End Git Integration Tests
# =========================================================================

def test_context_ingester_scenario_a(git_test_repo):
    """Test ContextIngester on a real Git repository with Scenario A branches."""
    repo_path = git_test_repo["repo_path"]
    ingester = ContextIngester(repo_path)

    context = ingester.ingest(
        branch_a=git_test_repo["branch_a"],
        branch_b=git_test_repo["branch_b"],
        file_path=git_test_repo["file_path"],
    )

    # 1. Base verification
    assert context.file_path == "process.py"
    assert context.base.commit_hash == git_test_repo["base_sha"]
    assert "def process(data):" in context.base.content
    assert "strict" not in context.base.content

    # 2. Branch A (ours) verification
    assert "strict=False" in context.ours.file_version.content
    assert len(context.ours.commit_log) > 0
    assert "strict flag" in context.ours.commit_log[0]
    assert "strict=False" in context.ours.diff_from_base
    ours_changed_names = [fc.name for fc in context.ours.changed_functions]
    assert "process" in ours_changed_names

    # Check signature_changed flag on Branch A's process
    proc_fc_ours = next(fc for fc in context.ours.changed_functions if fc.name == "process")
    assert proc_fc_ours.signature_changed is True

    # 3. Branch B (theirs) verification
    assert "data is None" in context.theirs.file_version.content
    assert len(context.theirs.commit_log) > 0
    assert "Bugfix" in context.theirs.commit_log[0]
    proc_fc_theirs = next(fc for fc in context.theirs.changed_functions if fc.name == "process")
    assert proc_fc_theirs.signature_changed is False
    assert proc_fc_theirs.body_changed is True

    # 4. Shared changed functions: both branches touched 'process' and 'summarize'
    assert "process" in context.shared_changed_functions
    assert "summarize" in context.shared_changed_functions

    # 5. AST diff summary
    assert "ours" in context.ast_diff_summary
    assert "theirs" in context.ast_diff_summary
    assert context.ast_diff_summary["ours"]["process"] == "modified"
    assert context.ast_diff_summary["theirs"]["process"] == "modified"


def test_context_ingester_conflict_markers_detection(git_test_repo):
    """Test that ContextIngester extracts working copy conflict markers when a merge conflict occurs."""
    repo_path = git_test_repo["repo_path"]

    # Trigger a real git merge conflict between branch_a and branch_b
    subprocess.run(["git", "-C", repo_path, "checkout", "branch_a"], check=True)
    subprocess.run(["git", "-C", repo_path, "merge", "branch_b"], check=False)

    ingester = ContextIngester(repo_path)
    context = ingester.ingest("branch_a", "branch_b", "process.py")

    assert context.conflict_markers is not None
    assert "<<<<<<<" in context.conflict_markers
    assert ">>>>>>>" in context.conflict_markers

    # Also test ingest_all_conflicts
    all_conflicts = ingester.ingest_all_conflicts("branch_a", "branch_b")
    assert len(all_conflicts) == 1
    assert all_conflicts[0].file_path == "process.py"


def test_ingester_error_handling(git_test_repo):
    """Test IngestionError is raised for non-existent branch or file."""
    repo_path = git_test_repo["repo_path"]
    ingester = ContextIngester(repo_path)

    with pytest.raises(IngestionError):
        ingester.ingest("branch_a", "non_existent_branch", "process.py")

    with pytest.raises(IngestionError):
        ingester.ingest("branch_a", "branch_b", "non_existent_file.py")


# =========================================================================
# 6. Property-Based Testing with Hypothesis
# =========================================================================

from hypothesis import given, strategies as st

@given(
    fn_name=st.from_regex(r"^[a-z][a-z0-9_]{0,10}$", fullmatch=True),
    args=st.lists(st.from_regex(r"^[a-z][a-z0-9_]{0,10}$", fullmatch=True), min_size=0, max_size=4, unique=True),
)
def test_hypothesis_function_signature_extraction(fn_name, args):
    """Property-based test: function extraction and signature parsing for arbitrary valid signatures."""
    code = f"def {fn_name}({', '.join(args)}):\n    return None\n"
    fns = extract_functions(code)
    assert fn_name in fns
    sig = get_signature(fns[fn_name])
    assert sig["name"] == fn_name
    assert sig["args"] == args

