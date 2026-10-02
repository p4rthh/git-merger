# git-merger

**Semantic Merge-Conflict Resolver with Differential Verification**

Autonomous coding agent for resolving Git merge conflicts using AST analysis, LLM semantic synthesis, and parallel differential test verification in isolated sandboxes.

## Architecture

- **Phase 1: Ingestion & AST Extraction (`src/git_merger/ingester.py`, `ast_analysis.py`, `git_ops.py`)**
  - Git 3-way merge context ingestion (`MergeContext`, `BranchContext`, `FileVersion`).
  - AST function extraction, signature comparison, call graph construction, and blast radius isolation.
- **Phase 2: Model Routing & Synthesis (`src/git_merger/synthesizer.py`, `model_router.py`)**
- **Phase 3: Sandbox Verification (`src/git_merger/sandbox.py`, `evaluator.py`)**
- **Phase 4: Closed-Loop Self-Healing (`src/git_merger/repair_loop.py`)**

## Installation & Testing

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest -v
```
