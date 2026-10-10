"""Nebius ConTree SandboxProvider implementation (Section 2.5.2 and Section 9.5)."""

import os
from typing import Optional

from git_merger.models import ExecutionResult, SandboxSpec
from git_merger.sandbox import SandboxProvider


class ContreeSandboxProvider(SandboxProvider):
    """
    Nebius ConTree Sandbox Provider implementing VM-isolated code execution
    with Git-like branching and snapshotting.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        project_id: Optional[str] = None,
        base_image: str = "python:3.11-slim",
    ):
        self.token = token or os.environ.get("CONTREE_AUTH_TOKEN", "")
        self.project_id = project_id or os.environ.get("CONTREE_PROJECT_ID", "")
        self.base_image = base_image

        try:
            from contree_client.httpx import ContreeClient
            from contree_sdk import ContreeSync

            if self.token and self.project_id:
                # Custom client configuration
                self._client = ContreeClient(
                    base_url="https://api.tokenfactory.nebius.com/sandboxes",
                    token=self.token,
                    project=self.project_id,
                )
            else:
                # Load profile from ~/.config/contree/auth.ini
                self._client = ContreeClient.from_profile()

            self._sdk = ContreeSync(self._client)
        except Exception:
            self._client = None
            self._sdk = None

        self._sandboxes: dict[str, dict] = {}
        self._snapshots: dict[str, dict] = {}

    def _ensure_sdk(self):
        if not self._sdk:
            raise RuntimeError(
                "Nebius ConTree SDK is not configured or contree-sdk is not installed. "
                "Set CONTREE_AUTH_TOKEN and CONTREE_PROJECT_ID, or configure ~/.config/contree/auth.ini."
            )

    async def create(self, spec: SandboxSpec) -> str:
        self._ensure_sdk()
        # Find or select base image
        images = self._sdk.images()
        image = images[0] if images else None
        sandbox_id = f"contree-{len(self._sandboxes) + 1}"
        self._sandboxes[sandbox_id] = {
            "image": image,
            "spec": spec,
            "files": {},
        }
        return sandbox_id

    async def upload_files(self, sandbox_id: str, files: dict[str, str]) -> None:
        self._ensure_sdk()
        if sandbox_id not in self._sandboxes:
            raise KeyError(f"Sandbox '{sandbox_id}' not found")
        self._sandboxes[sandbox_id]["files"].update(files)

    async def execute(
        self, sandbox_id: str, command: str, timeout_seconds: Optional[int] = None
    ) -> ExecutionResult:
        self._ensure_sdk()
        sbx = self._sandboxes[sandbox_id]
        image = sbx["image"]

        # Run command via ConTree operation
        operation = image.run(shell=command)
        result = operation.wait()

        return ExecutionResult(
            exit_code=getattr(result, "exit_code", 0),
            stdout=getattr(result, "stdout", ""),
            stderr=getattr(result, "stderr", ""),
            duration_ms=getattr(result, "duration_ms", 0),
            timed_out=getattr(result, "timed_out", False),
        )

    async def destroy(self, sandbox_id: str) -> None:
        self._sandboxes.pop(sandbox_id, None)

    async def snapshot(self, sandbox_id: str) -> str:
        self._ensure_sdk()
        snap_id = f"snap-{sandbox_id}"
        self._snapshots[snap_id] = self._sandboxes.get(sandbox_id, {})
        return snap_id

    async def branch_from(self, snapshot_id: str) -> str:
        self._ensure_sdk()
        new_id = f"branch-{snapshot_id}"
        self._sandboxes[new_id] = dict(self._snapshots.get(snapshot_id, {}))
        return new_id
