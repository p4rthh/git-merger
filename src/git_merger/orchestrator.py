
"""Top-level orchestration for semantic Git merge-conflict resolution."""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from openai import OpenAI

from git_merger.evaluator import DifferentialEvaluator
from git_merger.harness import HarnessGenerator
from git_merger.ingester import ContextIngester
from git_merger.models import EvaluationVerdict
from git_merger.repair import RepairSynthesizer
from git_merger.repair_loop import IterativeRepairLoop, RepairAttempt, TokenBudget
from git_merger.sandbox_manager import SandboxManager
from git_merger.synthesizer import SemanticSynthesizer
from git_merger.state import AgentState, ResolutionReport, StateLogger


logger = logging.getLogger("git_merger.orchestrator")


def _create_model_client() -> OpenAI:
    """Create the Nebius Token Factory OpenAI-compatible client."""
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NEBIUS_API_KEY is missing. Add it to your .env file "
            "and load that file before running the orchestrator."
        )

    return OpenAI(
        base_url="https://api.studio.nebius.ai/v1",
        api_key=api_key,
    )


def _commit_verified_candidate(
    repo_path: str,
    file_path: str,
    candidate_code: str,
) -> str:
    """Write, stage, and commit the verified resolution."""
    repo = Path(repo_path).resolve()
    target = (repo / file_path).resolve()

    # Prevent a file path from escaping the repository directory.
    if target != repo and repo not in target.parents:
        raise ValueError("file_path must point inside repo_path.")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(candidate_code, encoding="utf-8")

    subprocess.run(
        ["git", "add", "--", str(target.relative_to(repo))],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    subprocess.run(
        [
            "git",
            "-c", "user.name=git-merger",
            "-c", "user.email=git-merger@localhost",
            "commit",
            "-m", f"Resolve semantic merge conflict in {file_path}",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


async def resolve_merge_conflict(
    repo_path: str,
    branch_a: str,
    branch_b: str,
    file_path: str,
    max_repair_iterations: int = 5,
    *,
    model_client: OpenAI | None = None,
    sandbox_provider: Any | None = None,
) -> ResolutionReport:
    """
    Resolve a semantic merge conflict, verify the candidate, and commit it
    only if differential evaluation passes.

    Pass model_client and sandbox_provider explicitly when testing or when
    using a particular sandbox backend.
    """
    if max_repair_iterations < 0:
        raise ValueError("max_repair_iterations cannot be negative.")

    state_log = StateLogger()
    token_budget = TokenBudget(max_total_tokens=500_000)
    repair_history: list[RepairAttempt] = []
    candidate_code: str | None = None
    verdict: EvaluationVerdict | None = None
    provenance: dict[str, Any] = {}

    def escalate(reason: str) -> ResolutionReport:
        if state_log.state != AgentState.ESCALATING_TO_HUMAN:
            state_log.transition(AgentState.ESCALATING_TO_HUMAN, reason)

        return ResolutionReport(
            status="escalated",
            reason=reason,
            partial_candidate=candidate_code,
            provenance=provenance,
            repair_history=repair_history,
            total_iterations=len(repair_history),
            token_usage=token_budget.summary(),
            verdict=verdict,
            state_history=list(state_log.events),
        )

    try:
        # Phase 1: ingest the three-way merge context.
        state_log.transition(
            AgentState.INGESTING,
            "conflict_detected",
        )
        context = ContextIngester(repo_path).ingest(
            branch_a,
            branch_b,
            file_path,
        )
    except Exception as exc:
        logger.exception("Context ingestion failed.")
        return escalate(f"Context ingestion failed: {exc}")

    try:
        client = model_client or _create_model_client()

        # Phase 2: synthesize the initial candidate.
        state_log.transition(
            AgentState.SYNTHESIZING,
            "context_ready",
        )
        synthesizer = SemanticSynthesizer(client)
        synthesis = synthesizer.synthesize(context)

        candidate_code = synthesis.candidate_code
        provenance = synthesis.per_function_provenance or {}
        token_budget.record(synthesis.token_usage)

        # Phase 3: generate a differential test harness.
        state_log.transition(
            AgentState.GENERATING_HARNESS,
            "candidate_produced",
        )
        harness_generator = HarnessGenerator(client)
        harness = harness_generator.generate(context, candidate_code)
        token_budget.record(
            getattr(harness_generator, "last_token_usage", {})
        )

        # The provider is deliberately injected: this avoids silently
        # selecting a sandbox backend that may not be configured.
        if sandbox_provider is None:
            return escalate(
                "No sandbox_provider supplied. Pass a configured "
                "sandbox provider to run differential evaluation."
            )

        sandbox_manager = SandboxManager(sandbox_provider)
        evaluator = DifferentialEvaluator()

        # Phase 4: evaluate the initial candidate before attempting repairs.
        state_log.transition(
            AgentState.SANDBOX_EVALUATING,
            "harness_ready",
        )
        execution_results = await sandbox_manager.execute_differential(
            context,
            candidate_code,
            harness,
        )
        verdict = evaluator.evaluate(execution_results, context)

        if not verdict.passed:
            # RepairSynthesizer owns the .repair(...) method used by the
            # IterativeRepairLoop implementation.
            repair_synthesizer = RepairSynthesizer(client)
            repair_loop = IterativeRepairLoop(
                synthesizer=repair_synthesizer,
                harness_generator=harness_generator,
                sandbox_manager=sandbox_manager,
                evaluator=evaluator,
                max_iterations=max_repair_iterations,
                token_budget=token_budget,
            )

            state_log.transition(
                AgentState.REPAIRING,
                "tests_failed_can_retry",
            )
            candidate_code, verdict, repair_history = await repair_loop.run(
                context=context,
                initial_candidate=candidate_code,
                initial_verdict=verdict,
            )

            # The repair loop may have stopped because a genuine conflict
            # needs human judgment or because its token budget ran out.
            if repair_loop.escalation_reason:
                return escalate(repair_loop.escalation_reason)

            # Record repair-loop activity in the outer state log.
            for attempt in repair_history:
                state_log.transition(
                    AgentState.SANDBOX_EVALUATING,
                    f"repair_iteration_{attempt.iteration}_evaluated",
                )
                if not verdict.passed and attempt != repair_history[-1]:
                    state_log.transition(
                        AgentState.REPAIRING,
                        f"repair_iteration_{attempt.iteration}_continuing",
                    )

        if verdict is None or not verdict.passed:
            reason = (
                verdict.summary
                if verdict is not None
                else "No evaluation verdict was produced."
            )
            return escalate(reason)

        # Phase 5: commit only the verified candidate.
        state_log.transition(AgentState.SUCCESS, "all_tests_passed")

        commit_sha = _commit_verified_candidate(
            repo_path,
            file_path,
            candidate_code,
        )
        state_log.transition(AgentState.COMMITTED, "commit_created")

        pr_body = ""
        try:
            from git_merger.provenance import ProvenanceReporter
            reporter = ProvenanceReporter()
            prov_obj = reporter.build_provenance(
                file_path=file_path,
                per_function_provenance=provenance,
                verdict=verdict,
                repair_history=repair_history,
                token_usage=token_budget.summary(),
            )
            pr_body = reporter.generate_markdown(prov_obj)
        except Exception as prov_err:
            logger.warning("Failed to generate provenance markdown: %s", prov_err)

        return ResolutionReport(
            status="success",
            final_candidate=candidate_code,
            commit_sha=commit_sha,
            provenance=provenance,
            repair_history=repair_history,
            total_iterations=len(repair_history),
            token_usage=token_budget.summary(),
            pr_body=pr_body,
            verdict=verdict,
            state_history=list(state_log.events),
        )

    except Exception as exc:
        logger.exception("Merge conflict resolution failed.")
        return escalate(f"Resolution pipeline failed: {exc}")
