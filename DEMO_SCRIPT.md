# 🎙️ git-merger: 3-Minute Hackathon Demo Script

> **Target Audience:** Hackathon Judges, Principal Engineers, Open Source Maintainers  
> **Total Duration:** 3 Minutes (180 seconds)  
> **Setup Required:** Terminal open with `git-merger` installed and `.venv` active.

---

## ⏱️ Timeline & Stage Directions

```
 0:00 ── 0:30  [The Hook] Merge conflicts are semantic, not textual
 0:30 ── 1:15  [The Architecture] AST Analysis + LLM Synthesis + Differential Sandboxing
 1:15 ── 2:15  [Live Demo] Resolving Scenarios A, B, and C in real-time
 2:15 ── 2:45  [Explainability] PR Provenance Report & Zero-Regression Proof
 2:45 ── 3:00  [The Closing] Drop-in Git merge driver & future outlook
```

---

## Part 1: The Hook (0:00 – 0:30)

**[Visual: Presenter speaking directly or showing traditional Git conflict markers `<<<<<<<` `=======` `>>>>>>>`]**

> *"Every developer knows the dread of this screen: Git merge conflict markers. But here's the reality: Git's 3-way merge was invented in the 1980s. It operates strictly on text lines with zero understanding of programming language semantics.*
>
> *When Branch A updates a function signature to add a feature, and Branch B adds critical input validation to that same function, Git gives up and forces a human to manually merge lines. Even worse: when Git doesn't conflict, it often merges cleanly while silently breaking runtime behavior.*
>
> *Today, we introduce **git-merger**: the autonomous semantic conflict resolver with parallel differential verification."*

---

## Part 2: The Breakthrough Architecture (0:30 – 1:15)

**[Visual: Architecture diagram / System Overview]**

> *"Unlike naive AI coding assistants that simply ask an LLM to 'fix this conflict', `git-merger` implements a rigorous 5-phase scientific pipeline:*
>
> 1. *First, **AST Ingestion & Blast Radius Analysis**: We parse the abstract syntax tree of the base ancestor, Branch A, and Branch B. We extract call graphs and pinpoint exactly which functions and callers are impacted.*
> 2. *Second, **Semantic Synthesis**: Powered by NVIDIA Nemotron-3 Ultra via Nebius Token Factory, we synthesize a unified candidate that preserves the architectural intent of both branches.*
> 3. *Third, **Parallel Differential Sandboxing**: We dynamically generate property-based test harnesses using Hypothesis and dispatch them simultaneously into 4 isolated sandboxes: Base, Ours, Theirs, and Candidate.*
> 4. *Fourth, **Closed-Loop Self-Healing**: If any behavioral invariant from either branch is violated or a regression is detected, our repair loop feeds the pytest tracebacks back into the model to heal the candidate.*
> 5. *Finally, **Verified Commit**: We only commit code when differential verification is 100% green."*

---

## Part 3: Live Demo Walkthrough (1:15 – 2:15)

**[Action: Switch to terminal window and run:]**
```bash
git-merger demo
```

**[Visual: Fast colored terminal output showcasing state transitions, git repo creation, and resolution]**

> *"Let's watch `git-merger` solve real benchmark scenarios live.*
>
> *In **Scenario A**, Branch A evolved `process(data, strict=False)` with an optional argument. Branch B concurrently introduced type and nil-check validation guards. Notice the state machine moving in milliseconds:*
> `[INGESTING] -> [SYNTHESIZING] -> [GENERATING_HARNESS] -> [SANDBOX_EVALUATING] -> [COMMITTED]`.
>
> *Look at the final merged code: `git-merger` didn't choose between Branch A and Branch B. It synthesized both: it retained the new `strict=False` signature AND adapted the nil-checks to work inside the new signature.*
>
> *In **Scenario B**, two branches concurrently extended a service class with telemetry and rate limiting. `git-merger` resolved the class methods without dropping either branch's decorator or methods.*
>
> *In **Scenario C**, an algorithmic refactoring inverted loop logic while the other branch modified the return schema. All differential tests passed with 100% behavioral parity across all 4 sandboxes in under 2 seconds."*

---

## Part 4: Explainability & PR Provenance (2:15 – 2:45)

**[Visual: Scroll to the Provenance Report output in terminal or show Markdown preview]**

> *"How do developers and tech leads trust an autonomous agent? With **Explainable Resolution Provenance**.*
>
> *`git-merger` automatically posts a Markdown report directly into the GitHub PR body. It breaks down:*
> - *Every changed function with line-level rationale.*
> - *Exactly what was incorporated from Branch A and Branch B.*
> - *The differential test verdict proving zero behavioral regressions.*
> - *Full repair cycle history and resource token accounting.*
>
> *Engineers don't have to guess why a line was written—the report explains the semantic synthesis rationale with full traceability."*

---

## Part 5: The Closing & Next Steps (2:45 – 3:00)

**[Visual: Run `scripts/install_merge_driver.sh` or show `.gitconfig`]**

> *"Best of all, `git-merger` isn't a complex web portal or proprietary IDE—it integrates directly into your existing workflow as a native Git merge driver with a single shell command:*
>
> ```bash
> ./scripts/install_merge_driver.sh
> ```
>
> *Whenever you run `git merge` or `git pull`, `git-merger` handles Python conflicts silently, safely, and verifiably in the background.*
>
> *Thank you, and we welcome your questions!"*
