"""AST parsing, structural diffing, call-graph analysis, and blast-radius calculation."""

import ast
from typing import Optional, Union

from git_merger.exceptions import ASTAnalysisError
from git_merger.models import BlastRadius, FunctionChange

FunctionNode = Union[ast.FunctionDef, ast.AsyncFunctionDef]


def extract_functions(source: str) -> dict[str, FunctionNode]:
    """
    Parse Python source code and return a dictionary of function_name -> AST node.
    Includes both synchronous and asynchronous function definitions.

    Args:
        source: Python source code string.

    Returns:
        dict mapping function name to its AST node.

    Raises:
        ASTAnalysisError: If the source code cannot be parsed.
    """
    if not source or not source.strip():
        return {}

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ASTAnalysisError(f"Syntax error parsing source code: {e}") from e

    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def get_function_source(source: str, node: FunctionNode) -> str:
    """
    Extract the source code slice of a specific function node.
    Falls back to ast.unparse() if the source segment cannot be retrieved.

    Args:
        source: Full Python source code string.
        node: The function AST node.

    Returns:
        Source code of the function.
    """
    try:
        segment = ast.get_source_segment(source, node)
        if segment is not None:
            return segment
    except Exception:
        pass

    try:
        return ast.unparse(node)
    except Exception as e:
        raise ASTAnalysisError(f"Failed to unparse function '{node.name}': {e}") from e


def get_signature(node: FunctionNode) -> dict:
    """
    Extract function signature as a comparable dictionary.
    Follows Section 2.2 of PROJECT_SPEC.md.

    Args:
        node: FunctionDef or AsyncFunctionDef node.

    Returns:
        Comparable dictionary of function signature components.
    """
    return {
        "name": node.name,
        "args": [a.arg for a in node.args.args],
        "defaults": [ast.dump(d) for d in node.args.defaults],
        "posonlyargs": [a.arg for a in getattr(node.args, "posonlyargs", [])],
        "kwonly": [a.arg for a in node.args.kwonlyargs],
        "kw_defaults": [ast.dump(d) if d else None for d in node.args.kw_defaults],
        "vararg": node.args.vararg.arg if node.args.vararg else None,
        "kwarg": node.args.kwarg.arg if node.args.kwarg else None,
        "returns": ast.dump(node.returns) if node.returns else None,
        "decorators": [ast.dump(d) for d in node.decorator_list],
    }


def diff_functions(base_src: str, branch_src: str) -> dict[str, str]:
    """
    Compare function definitions between base and branch source code.
    Follows Section 2.2 of PROJECT_SPEC.md.

    Args:
        base_src: Base version Python source code.
        branch_src: Branch version Python source code.

    Returns:
        dict mapping function name to change status: "added", "removed", or "modified".
    """
    base_fns = extract_functions(base_src)
    branch_fns = extract_functions(branch_src)
    result = {}

    for name in set(base_fns) | set(branch_fns):
        if name not in base_fns:
            result[name] = "added"
        elif name not in branch_fns:
            result[name] = "removed"
        elif ast.dump(base_fns[name]) != ast.dump(branch_fns[name]):
            result[name] = "modified"

    return result


def analyze_function_change(
    name: str,
    base_node: Optional[FunctionNode],
    branch_node: Optional[FunctionNode],
    base_src: Optional[str] = None,
    branch_src: Optional[str] = None,
) -> Optional[FunctionChange]:
    """
    Perform fine-grained analysis of changes to a function between base and branch.

    Returns:
        FunctionChange instance or None if unchanged.
    """
    if base_node is None and branch_node is None:
        return None

    if base_node is None:
        # Added in branch
        branch_source = get_function_source(branch_src, branch_node) if branch_src else ast.unparse(branch_node)
        return FunctionChange(
            name=name,
            change_type="added",
            base_source=None,
            branch_source=branch_source,
            signature_changed=True,
            body_changed=True,
        )

    if branch_node is None:
        # Removed in branch
        base_source = get_function_source(base_src, base_node) if base_src else ast.unparse(base_node)
        return FunctionChange(
            name=name,
            change_type="removed",
            base_source=base_source,
            branch_source=None,
            signature_changed=True,
            body_changed=True,
        )

    # Both exist: check signature and body differences
    sig_base = get_signature(base_node)
    sig_branch = get_signature(branch_node)
    sig_changed = sig_base != sig_branch

    body_base_dump = [ast.dump(stmt) for stmt in base_node.body]
    body_branch_dump = [ast.dump(stmt) for stmt in branch_node.body]
    body_changed = body_base_dump != body_branch_dump

    if not sig_changed and not body_changed:
        return None

    base_source = get_function_source(base_src, base_node) if base_src else ast.unparse(base_node)
    branch_source = get_function_source(branch_src, branch_node) if branch_src else ast.unparse(branch_node)

    return FunctionChange(
        name=name,
        change_type="modified",
        base_source=base_source,
        branch_source=branch_source,
        signature_changed=sig_changed,
        body_changed=body_changed,
    )


def build_call_graph(source: str) -> dict[str, set[str]]:
    """
    Walk the AST and for each FunctionDef, find all calls to other functions
    defined in the same module. Follows Section 7.1 of PROJECT_SPEC.md.

    Args:
        source: Python source code string.

    Returns:
        dict mapping caller_function_name -> set of callee_function_names.
    """
    if not source or not source.strip():
        return {}

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ASTAnalysisError(f"Syntax error building call graph: {e}") from e

    functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    call_graph: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            callees = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name) and child.func.id in functions:
                        callees.add(child.func.id)
                    elif isinstance(child.func, ast.Attribute):
                        if child.func.attr in functions:
                            callees.add(child.func.attr)
            call_graph[node.name] = callees

    return call_graph


def compute_transitive_callers(
    call_graph: dict[str, set[str]],
    targets: set[str],
) -> set[str]:
    """
    Compute all functions that transitively call any function in the target set.
    Follows Section 7.1 of PROJECT_SPEC.md.

    Args:
        call_graph: dict mapping caller -> set of callees.
        targets: set of target function names.

    Returns:
        set of all functions that directly or transitively call any target.
    """
    # Invert the call graph: callee -> set of callers
    callers_of: dict[str, set[str]] = {}
    for caller, callees in call_graph.items():
        for callee in callees:
            callers_of.setdefault(callee, set()).add(caller)

    affected = set(targets)
    queue = list(targets)
    while queue:
        current = queue.pop(0)
        for caller in callers_of.get(current, set()):
            if caller not in affected:
                affected.add(caller)
                queue.append(caller)

    return affected


class BlastRadiusAnalyzer:
    """
    AST-Guided Blast Radius Isolation (Section 7.1).
    Pre-filters test generation strictly to functions whose AST signatures
    or call graphs changed.
    """

    def analyze(
        self,
        base_src: str,
        ours_src: str,
        theirs_src: str,
    ) -> BlastRadius:
        """
        Analyze AST changes between base and both branches, build call graphs,
        and compute the full transitive blast radius.

        Args:
            base_src: Common ancestor Python source code.
            ours_src: Branch A Python source code.
            theirs_src: Branch B Python source code.

        Returns:
            BlastRadius with directly_changed, callers, affected, and unchanged sets.
        """
        # 1. Identify directly changed functions
        diff_ours = diff_functions(base_src, ours_src)
        diff_theirs = diff_functions(base_src, theirs_src)
        directly_changed = set(diff_ours.keys()) | set(diff_theirs.keys())

        # 2. Build combined call graph across all three versions
        call_graph_base = build_call_graph(base_src)
        call_graph_ours = build_call_graph(ours_src)
        call_graph_theirs = build_call_graph(theirs_src)

        all_callers = set(call_graph_base.keys()) | set(call_graph_ours.keys()) | set(call_graph_theirs.keys())
        unified_call_graph: dict[str, set[str]] = {}
        for caller in all_callers:
            callees = (
                call_graph_base.get(caller, set())
                | call_graph_ours.get(caller, set())
                | call_graph_theirs.get(caller, set())
            )
            unified_call_graph[caller] = callees

        # Build callers map: func_name -> set of functions that call it
        callers_map: dict[str, set[str]] = {}
        for caller, callees in unified_call_graph.items():
            for callee in callees:
                callers_map.setdefault(callee, set()).add(caller)

        # 3. Transitive closure of callers for directly changed functions
        affected = compute_transitive_callers(unified_call_graph, directly_changed)

        # 4. Determine unchanged functions
        all_functions = (
            set(extract_functions(base_src).keys())
            | set(extract_functions(ours_src).keys())
            | set(extract_functions(theirs_src).keys())
        )
        unchanged = all_functions - affected

        return BlastRadius(
            directly_changed=directly_changed,
            callers=callers_map,
            affected=affected,
            unchanged=unchanged,
        )
