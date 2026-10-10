"""Sandbox Provider Abstraction layer (Section 2.5.3 and Section 9.5)."""

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from typing import Optional

from git_merger.models import ExecutionResult, SandboxSpec, SandboxStatus


class SandboxProvider(ABC):
    """Abstract Base Class for isolated sandbox execution environments."""

    @abstractmethod
    async def create(self, spec: SandboxSpec) -> str:
        """Provision a sandbox environment. Returns sandbox_id."""
        pass

    @abstractmethod
    async def upload_files(self, sandbox_id: str, files: dict[str, str]) -> None:
        """Upload a file tree into the sandbox. files = {relative_path: content}."""
        pass

    @abstractmethod
    async def execute(
        self, sandbox_id: str, command: str, timeout_seconds: Optional[int] = None
    ) -> ExecutionResult:
        """Execute a shell command inside the sandbox."""
        pass

    @abstractmethod
    async def destroy(self, sandbox_id: str) -> None:
        """Tear down and clean up the sandbox environment."""
        pass

    @abstractmethod
    async def snapshot(self, sandbox_id: str) -> str:
        """Create an immutable checkpoint/snapshot. Returns snapshot_id."""
        pass

    @abstractmethod
    async def branch_from(self, snapshot_id: str) -> str:
        """Fork a new sandbox environment from a snapshot. Returns sandbox_id."""
        pass


class MockSandboxProvider(SandboxProvider):
    """
    In-process, filesystem-isolated Sandbox Provider for deterministic testing.
    Uses temporary workspace directories and subprocess execution.
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or tempfile.gettempdir()
        self._sandboxes: dict[str, dict] = {}
        self._snapshots: dict[str, dict] = {}
        self._counter: int = 0

    async def create(self, spec: SandboxSpec) -> str:
        self._counter += 1
        sandbox_id = f"mock-sbx-{self._counter}-{int(time.time() * 1000)}"
        workdir = tempfile.mkdtemp(prefix=f"{sandbox_id}_", dir=self.base_dir)

        self._sandboxes[sandbox_id] = {
            "id": sandbox_id,
            "workdir": workdir,
            "spec": spec,
            "status": SandboxStatus.READY,
            "files": {},
        }
        return sandbox_id

    async def upload_files(self, sandbox_id: str, files: dict[str, str]) -> None:
        if sandbox_id not in self._sandboxes:
            raise KeyError(f"Sandbox '{sandbox_id}' not found")

        sbx = self._sandboxes[sandbox_id]
        workdir = sbx["workdir"]

        for rel_path, content in files.items():
            dest_path = os.path.join(workdir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(content)
            sbx["files"][rel_path] = content

    async def execute(
        self, sandbox_id: str, command: str, timeout_seconds: Optional[int] = None
    ) -> ExecutionResult:
        if sandbox_id not in self._sandboxes:
            raise KeyError(f"Sandbox '{sandbox_id}' not found")

        sbx = self._sandboxes[sandbox_id]
        workdir = sbx["workdir"]
        timeout = timeout_seconds or sbx["spec"].timeout_seconds

        start_time = time.monotonic()
        timed_out = False

        # Prepare environment
        env = os.environ.copy()
        python_bin_dir = os.path.dirname(sys.executable)
        env["PATH"] = python_bin_dir + os.pathsep + env.get("PATH", "")
        env["PYTHONPATH"] = workdir + os.pathsep + env.get("PYTHONPATH", "")

        if command.startswith("python "):
            command = f"{sys.executable} {command[7:]}"
        elif command == "python":
            command = sys.executable

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=workdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")
            exit_code = proc.returncode or 0
        except asyncio.TimeoutError:
            timed_out = True
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            stdout = ""
            stderr = f"Command timed out after {timeout} seconds"
            exit_code = 124

        duration_ms = int((time.monotonic() - start_time) * 1000)

        return ExecutionResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
            timed_out=timed_out,
        )

    async def destroy(self, sandbox_id: str) -> None:
        if sandbox_id in self._sandboxes:
            sbx = self._sandboxes.pop(sandbox_id)
            workdir = sbx["workdir"]
            if os.path.exists(workdir):
                shutil.rmtree(workdir, ignore_errors=True)

    async def snapshot(self, sandbox_id: str) -> str:
        if sandbox_id not in self._sandboxes:
            raise KeyError(f"Sandbox '{sandbox_id}' not found")

        sbx = self._sandboxes[sandbox_id]
        snap_id = f"snap-{sandbox_id}-{int(time.time() * 1000)}"
        snap_dir = tempfile.mkdtemp(prefix=f"{snap_id}_", dir=self.base_dir)

        # Copy existing workdir
        shutil.copytree(sbx["workdir"], snap_dir, dirs_exist_ok=True)
        self._snapshots[snap_id] = {
            "id": snap_id,
            "workdir": snap_dir,
            "spec": sbx["spec"],
            "files": dict(sbx["files"]),
        }
        return snap_id

    async def branch_from(self, snapshot_id: str) -> str:
        if snapshot_id not in self._snapshots:
            raise KeyError(f"Snapshot '{snapshot_id}' not found")

        snap = self._snapshots[snapshot_id]
        self._counter += 1
        new_sbx_id = f"mock-branch-{self._counter}-{int(time.time() * 1000)}"
        new_workdir = tempfile.mkdtemp(prefix=f"{new_sbx_id}_", dir=self.base_dir)

        shutil.copytree(snap["workdir"], new_workdir, dirs_exist_ok=True)
        self._sandboxes[new_sbx_id] = {
            "id": new_sbx_id,
            "workdir": new_workdir,
            "spec": snap["spec"],
            "status": SandboxStatus.READY,
            "files": dict(snap["files"]),
        }
        return new_sbx_id


def get_sandbox_provider(provider_type: Optional[str] = None) -> SandboxProvider:
    """
    Factory function returning the configured SandboxProvider instance.
    Defaults to MockSandboxProvider when no cloud token is provided or during tests.
    """
    p_type = (provider_type or os.environ.get("SANDBOX_PROVIDER", "mock")).lower()

    if p_type in ("contree", "nebius"):
        try:
            from git_merger.sandbox_contree import ContreeSandboxProvider
            return ContreeSandboxProvider()
        except ImportError:
            return MockSandboxProvider()
    elif p_type in ("docker", "local_docker"):
        try:
            from git_merger.sandbox_docker import LocalDockerProvider
            return LocalDockerProvider()
        except ImportError:
            return MockSandboxProvider()
    else:
        return MockSandboxProvider()
