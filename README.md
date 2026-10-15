# git-merger

> **Semantic Merge-Conflict Resolver with Differential Verification**  
> Autonomous coding agent that resolves Git merge conflicts using AST analysis, LLM semantic synthesis, and parallel differential test verification in isolated execution sandboxes.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 📖 Overview

Standard Git 3-way merge algorithms (`diff3`, `ort`) operate purely on textual lines without comprehension of code semantics, function call chains, or runtime contracts. When two branches modify overlapping lines or interrelated functions, Git either produces messy conflict markers (`<<<<<<<` / `>>>>>>>`) or silently accepts textual merges that break application behavior.

**git-merger** solves this with an autonomous 5-phase agent pipeline:
1. **AST & Call-Graph Ingestion**: Parses the common ancestor (`base`), current branch (`ours`), and incoming branch (`theirs`). Determines blast radius and affected call hierarchies.
2. **Semantic Intent Synthesis**: Employs **NVIDIA Nemotron-3 Ultra (550B)** via Nebius Token Factory to generate a unified candidate $M^*$ preserving the architectural intent of both branches.
3. **Property-Based Differential Test Generation**: Generates Hypothesis test suites capturing behavioral invariants from both sides.
4. **Parallel Differential Sandbox Verification**: Dispatches test harnesses concurrently across 4 isolated sandboxes: Base, Ours, Theirs, and Candidate $M^*$.
5. **Closed-Loop Self-Healing**: Feeds test failure traces into an iterative repair loop. Code is committed **only** when differential testing proves zero behavioral regressions.

---

## 🏗️ Architecture & Component Topology

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        git-merger Agent Pipeline                        │
│                                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐ │
│  │   Git    │───▶│   Context    │───▶│  Semantic   │───▶│  Harness   │ │
│  │ CLI/Hook │    │  Ingester    │    │ Synthesizer │    │ Generator  │ │
│  └──────────┘    └──────────────┘    └─────────────┘    └────────────┘ │
│                         │                   │                  │        │
│                         ▼                   ▼                  ▼        │
│                  ┌──────────────┐    ┌─────────────┐    ┌────────────┐ │
│                  │  MergeContext│    │ Candidate   │    │   Test     │ │
│                  │  (data obj)  │    │  M* code    │    │  Harness   │ │
│                  └──────┬───────┘    └──────┬──────┘    └─────┬──────┘ │
│                         │                   │                  │        │
│                         ▼                   ▼                  ▼        │
│                  ┌─────────────────────────────────────────────────┐    │
│                  │            Sandbox Manager                      │    │
│                  │  ┌─────────┐  ┌─────────┐  ┌───────────────┐  │    │
│                  │  │Sandbox A│  │Sandbox B│  │ Sandbox M*    │  │    │
│                  │  │(ours)   │  │(theirs) │  │ (candidate)   │  │    │
│                  │  └────┬────┘  └────┬────┘  └──────┬────────┘  │    │
│                  │       │            │               │           │    │
│                  └───────┼────────────┼───────────────┼───────────┘    │
│                          ▼            ▼               ▼                │
│                  ┌─────────────────────────────────────────────┐       │
│                  │       Differential Evaluator                │       │
│                  │  Compare results: A vs M*, B vs M*          │       │
│                  └────────────────────┬────────────────────────┘       │
│                                       │                                │
│                              ┌────────┴────────┐                      │
│                              ▼                  ▼                      │
│                       ┌───────────┐      ┌─────────────┐              │
│                       │  SUCCESS  │      │   Repair    │              │
│                       │  Commit & │      │   Loop      │──────┐      │
│                       │  PR Gen   │      │  (Nemotron  │      │      │
│                       └───────────┘      │   Ultra)    │      │      │
│                                          └─────────────┘      │      │
│                                                ▲              │      │
│                                                └──────────────┘      │
│                                          (max 5 iterations)          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites
- Python 3.11 or higher
- Git 2.30+

### Setup
```bash
# Clone the repository
git clone https://github.com/p4rthh/git-merger.git
cd git-merger

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e .
```

---

## ⚡ Quick-Start Guide

### 1. Configure Model API Key
Copy `.env.example` to `.env` and set your Nebius AI Studio token:
```bash
cp .env.example .env
# Edit .env and supply NEBIUS_API_KEY
```

### 2. Run Direct Resolution on a Repo
```bash
git-merger resolve \
  --repo-path /path/to/repo \
  --branch-a feature/payment-v2 \
  --branch-b feature/retry-logic \
  --file-path src/payments/service.py \
  --max-iterations 5
```

### 3. Native Git Merge Driver Integration
`git-merger` can act as your default Git custom merge driver for Python files:
```bash
# Install merge driver into local repo (or pass --global for all repos)
./scripts/install_merge_driver.sh

# Now, standard git merges automatically invoke git-merger!
git merge feature-branch
```

---

## 🎮 Demo Walkthrough

Run the built-in benchmark runner to test Scenarios A, B, and C with full live terminal visualization:
```bash
git-merger demo
```

### Included Benchmark Scenarios
| Scenario | Conflict Type | Characteristics |
|---|---|---|
| **Scenario A** | `signature_body` | Branch A evolves function signature; Branch B introduces nil-check and type guards. |
| **Scenario B** | `concurrent_extension` | Concurrent class extensions adding caching, rate-limiting, and metrics decorators. |
| **Scenario C** | `logic_inversion` | Branch A refactors loop/filtering algorithm; Branch B updates schema and error codes. |

---

## 🤖 Model Routing Configuration

`git-merger` uses specialized model routing via Nebius Token Factory:

| Pipeline Stage | Model Architecture | Temperature | Max Tokens |
|---|---|---|---|
| **Semantic Synthesis** | `nvidia/nemotron-3-ultra-550b-a55b` | 0.1 | 8,192 |
| **Harness Generation** | `nvidia/nemotron-3-super-120b-a12b` | 0.2 | 8,192 |
| **Log Parsing & AST Diff** | `nvidia/nemotron-3-nano-30b-a3b` | 0.0 | 4,096 |
| **Iterative Repair** | `nvidia/nemotron-3-ultra-550b-a55b` | 0.15 | 8,192 |

---

## 📊 Explainable Resolution Provenance (PR Body)

Whenever a conflict is resolved, `git-merger` emits a comprehensive Markdown report designed for GitHub Pull Request bodies:

```markdown
## 🔀 Semantic Merge Resolution Report

**File:** `data_processor.py`
**Conflict Type:** `signature_body`
**Confidence:** 98%
**Strategy:** AST-guided semantic synthesis combining Branch A signatures with Branch B validation

### Per-Function Provenance
| Function | Source | Changes from A | Changes from B | Rationale |
|---|---|---|---|---|
| `process` | `both` | Added `strict=False` | Nil-check guards | Adapted guards to work with new signature |

### Differential Verification
✅ **Verdict:** All 4 differential test sandboxes passed (base, ours, theirs, candidate).
- ✅ All behavioral changes from Branch A preserved
- ✅ All behavioral changes from Branch B preserved
- ✅ Zero regressions detected against base version
```

---

## 🧪 Testing & Verification

Run the comprehensive unit, property-based (Hypothesis), and end-to-end test suites:

```bash
# Run all tests
pytest -v

# Run Phase 5 specific tests (provenance & adaptive pruner)
pytest -v tests/test_provenance.py tests/test_pruner.py
```

---

## 🤝 Contributing

1. Fork the repository & create a feature branch (`git checkout -b feature/amazing-feature`).
2. Ensure all tests pass (`pytest -v`).
3. Commit your changes (`git commit -m 'feat: Add amazing feature'`).
4. Push to your branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
