"""Command dispatch module."""

from enum import Enum
from typing import Any
import os
import time


class Command(Enum):
    PING = "ping"
    ECHO = "echo"
    STATUS = "status"
    RESTART = "restart"
    HEALTH = "health"


def handle_ping(payload: dict) -> dict:
    return {"response": "pong", "timestamp": payload.get("timestamp")}


def handle_echo(payload: dict) -> dict:
    return {"response": payload.get("message", "")}


def handle_status(payload: dict) -> dict:
    """Enhanced status with uptime."""
    return {
        "response": "ok",
        "version": "1.1.0",
        "uptime_seconds": time.monotonic(),
    }


def handle_restart(payload: dict) -> dict:
    """Restart the service with optional delay."""
    delay = payload.get("delay_seconds", 0)
    force = payload.get("force", False)
    return {
        "response": "restarting",
        "delay": delay,
        "force": force,
        "pid": os.getpid(),
    }


def handle_health(payload: dict) -> dict:
    """Deep health check."""
    checks = {
        "memory": True,
        "disk": True,
        "services": payload.get("check_services", []),
    }
    return {
        "response": "healthy" if all([checks["memory"], checks["disk"]]) else "degraded",
        "checks": checks,
    }


DISPATCH_TABLE: dict[Command, callable] = {
    Command.PING: handle_ping,
    Command.ECHO: handle_echo,
    Command.STATUS: handle_status,
    Command.RESTART: handle_restart,
    Command.HEALTH: handle_health,
}


def dispatch(command: str, payload: dict) -> dict:
    """Dispatch a command string to the appropriate handler."""
    try:
        cmd = Command(command)
    except ValueError:
        return {"error": f"Unknown command: {command}"}
    handler = DISPATCH_TABLE[cmd]
    return handler(payload)
