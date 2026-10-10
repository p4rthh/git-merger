"""Sandbox Manager for parallel differential test orchestration (Section 3.2.4 and Section 5.3)."""

import asyncio
import os
import sys
from typing import Optional

from git_merger.models import (
    ExecutionResult,
    MergeContext,
    ParallelExecutionResult,
    SandboxSpec,
    TestHarness,
)
from git_merger.sandbox import SandboxProvider, get_sandbox_provider


RUN_TESTS_SCRIPT = """#!/bin/bash
set -e
export TARGET_MODULE="target_module"
python -m pytest test_differential.py -v --tb=long --no-header
"""


class SandboxManager:
    """
    Manages and orchestrates isolated sandbox environments for parallel
    execution of differential test harnesses across base, ours, theirs, and candidate versions.
    """

    def __init__(self, provider: Optional[SandboxProvider] = None):
        self.provider = provider or get_sandbox_provider()

    async def execute_differential(
        self,
        context: MergeContext,
        candidate_code: str,
        harness: TestHarness,
        spec: Optional[SandboxSpec] = None,
    ) -> ParallelExecutionResult:
        """
        Execute the test harness across 4 sandboxes concurrently:
        1. base version
        2. branch A (ours)
        3. branch B (theirs)
        4. candidate M*

        Returns:
            ParallelExecutionResult containing results from all 4 branches.
        """
        sandbox_spec = spec or SandboxSpec(
            python_version="3.11",
            memory_mb=512,
            timeout_seconds=harness.estimated_runtime_seconds or 60,
            pip_packages=["pytest", "hypothesis"] + (harness.requirements or []),
        )

        versions = {
            "base": context.base.content,
            "ours": context.ours.file_version.content,
            "theirs": context.theirs.file_version.content,
            "candidate": candidate_code,
        }

        # Build file trees for each sandbox
        tasks = {}
        for label, code in versions.items():
            file_tree = {
                "target_module.py": code,
                "test_differential.py": harness.test_code,
                "conftest.py": harness.conftest_code or "",
                "requirements.txt": "\n".join(sandbox_spec.pip_packages),
                "run_tests.sh": RUN_TESTS_SCRIPT,
            }
            tasks[label] = self._create_and_run(label, file_tree, sandbox_spec)

        # Execute concurrently across all 4 environments
        results_list = await asyncio.gather(
            tasks["base"],
            tasks["ours"],
            tasks["theirs"],
            tasks["candidate"],
            return_exceptions=True,
        )

        # Normalize any unexpected exceptions into failed ExecutionResults
        normalized_results = []
        for i, res in enumerate(results_list):
            if isinstance(res, Exception):
                normalized_results.append(
                    ExecutionResult(
                        exit_code=1,
                        stdout="",
                        stderr=f"Sandbox execution error: {res}",
                        duration_ms=0,
                        timed_out=False,
                    )
                )
            else:
                normalized_results.append(res)

        base_res, ours_res, theirs_res, candidate_res = normalized_results

        return ParallelExecutionResult(
            base_result=base_res,
            ours_result=ours_res,
            theirs_result=theirs_res,
            candidate_result=candidate_res,
            all_passed=all(r.exit_code == 0 for r in normalized_results),
        )

    async def _create_and_run(
        self,
        label: str,
        files: dict[str, str],
        spec: SandboxSpec,
    ) -> ExecutionResult:
        """Create sandbox, upload files, execute pytest, and destroy sandbox."""
        sandbox_id = await self.provider.create(spec)
        try:
            await self.provider.upload_files(sandbox_id, files)

            command = f"{sys.executable} -m pytest test_differential.py -v --tb=long --no-header"
            result = await self.provider.execute(
                sandbox_id=sandbox_id,
                command=command,
                timeout_seconds=spec.timeout_seconds,
            )
            return result
        finally:
            await self.provider.destroy(sandbox_id)
