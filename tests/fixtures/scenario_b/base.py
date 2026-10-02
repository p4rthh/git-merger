"""Command dispatch module."""

from enum import Enum
from typing import Any


class Command(Enum):
    PING = "ping"
    ECHO = "echo"
    STATUS = "status"


def handle_ping(payload: dict) -> dict:
    return {"response": "pong", "timestamp": payload.get("timestamp")}


def handle_echo(payload: dict) -> dict:
    return {"response": payload.get("message", "")}


def handle_status(payload: dict) -> dict:
    return {"response": "ok", "version": "1.0.0"}


DISPATCH_TABLE: dict[Command, callable] = {
    Command.PING: handle_ping,
    Command.ECHO: handle_echo,
    Command.STATUS: handle_status,
}


def dispatch(command: str, payload: dict) -> dict:
    """Dispatch a command string to the appropriate handler."""
    try:
        cmd = Command(command)
    except ValueError:
        return {"error": f"Unknown command: {command}"}
    handler = DISPATCH_TABLE[cmd]
    return handler(payload)
