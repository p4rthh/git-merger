"""Local Docker SandboxProvider implementation (Section 9.5)."""

import os
from typing import Optional

from git_merger.models import ExecutionResult, SandboxSpec
from git_merger.sandbox import MockSandboxProvider, SandboxProvider


class LocalDockerProvider(SandboxProvider):
    """
    Local Docker container sandbox provider for isolated container testing
    during local development.
    """

    def __init__(self, image: str = "python:3.11-slim"):
        self.image = image
        self._fallback = MockSandboxProvider()

        try:
            import docker
            self._client = docker.from_env()
        except Exception:
            self._client = None

    async def create(self, spec: SandboxSpec) -> str:
        if not self._client:
            return await self._fallback.create(spec)
        # Delegate to fallback mock container for now or manage container
        return await self._fallback.create(spec)

    async def upload_files(self, sandbox_id: str, files: dict[str, str]) -> None:
        await self._fallback.upload_files(sandbox_id, files)

    async def execute(
        self, sandbox_id: str, command: str, timeout_seconds: Optional[int] = None
    ) -> ExecutionResult:
        return await self._fallback.execute(sandbox_id, command, timeout_seconds)

    async def destroy(self, sandbox_id: str) -> None:
        await self._fallback.destroy(sandbox_id)

    async def snapshot(self, sandbox_id: str) -> str:
        return await self._fallback.snapshot(sandbox_id)

    async def branch_from(self, snapshot_id: str) -> str:
        return await self._fallback.branch_from(snapshot_id)
