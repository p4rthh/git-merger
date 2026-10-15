
"""State machine for the Git Merger agent.

Validates state transitions, records UTC timestamps, and emits JSON logs.
"""

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from git_merger.models import ResolutionReport

logger = logging.getLogger("git_merger.state")


class AgentState(str, Enum):
    IDLE = "IDLE"
    INGESTING = "INGESTING"
    SYNTHESIZING = "SYNTHESIZING"
    GENERATING_HARNESS = "GENERATING_HARNESS"
    SANDBOX_EVALUATING = "SANDBOX_EVALUATING"
    REPAIRING = "REPAIRING"
    SUCCESS = "SUCCESS"
    COMMITTED = "COMMITTED"
    ESCALATING_TO_HUMAN = "ESCALATING_TO_HUMAN"


# Current state -> event -> next state
TRANSITIONS: dict[AgentState, dict[str, AgentState]] = {
    AgentState.IDLE: {
        "conflict_detected": AgentState.INGESTING,
    },
    AgentState.INGESTING: {
        "context_ready": AgentState.SYNTHESIZING,
        "ingestion_error": AgentState.ESCALATING_TO_HUMAN,
    },
    AgentState.SYNTHESIZING: {
        "candidate_produced": AgentState.GENERATING_HARNESS,
        "synthesis_error": AgentState.ESCALATING_TO_HUMAN,
    },
    AgentState.GENERATING_HARNESS: {
        "harness_ready": AgentState.SANDBOX_EVALUATING,
        "harness_error": AgentState.ESCALATING_TO_HUMAN,
    },
    AgentState.SANDBOX_EVALUATING: {
        "all_tests_passed": AgentState.SUCCESS,
        "tests_failed_can_retry": AgentState.REPAIRING,
        "tests_failed_max_retries": AgentState.ESCALATING_TO_HUMAN,
        "sandbox_error": AgentState.ESCALATING_TO_HUMAN,
    },
    AgentState.REPAIRING: {
        "repair_produced": AgentState.SANDBOX_EVALUATING,
        "repair_declares_conflict": AgentState.ESCALATING_TO_HUMAN,
    },
    AgentState.SUCCESS: {
        "commit_created": AgentState.COMMITTED,
    },
    AgentState.COMMITTED: {},
    AgentState.ESCALATING_TO_HUMAN: {},
}


class AgentStateMachine:
    """Track agent states and validate transitions between them."""

    def __init__(
        self,
        initial_state: AgentState = AgentState.IDLE,
    ) -> None:
        self._current_state = AgentState(initial_state)
        self._history: list[dict[str, Any]] = []

    @property
    def current_state(self) -> AgentState:
        """Return the agent's current state."""
        return self._current_state

    def transition(self, event: str) -> AgentState:
        """Apply an event if it is valid for the current state.

        Raises:
            ValueError: If the event is not allowed from the current state.
        """
        from_state = self._current_state
        allowed_transitions = TRANSITIONS.get(from_state, {})
        to_state = allowed_transitions.get(event)

        if to_state is None:
            allowed_events = list(allowed_transitions.keys())
            message = (
                f"Invalid transition: state={from_state.value!r}, "
                f"event={event!r}. Allowed events: {allowed_events}"
            )

            # Log rejected transitions too, so errors are diagnosable.
            self._emit_log({
                "timestamp": self._timestamp(),
                "from_state": from_state.value,
                "event": event,
                "to_state": None,
                "valid": False,
                "error": message,
            })
            raise ValueError(message)

        record = {
            "timestamp": self._timestamp(),
            "from_state": from_state.value,
            "event": event,
            "to_state": to_state.value,
            "valid": True,
        }

        self._current_state = to_state
        self._history.append(record)
        self._emit_log(record)

        return self._current_state

    def history(self) -> list[dict[str, Any]]:
        """Return a copy of the recorded transition history."""
        return [record.copy() for record in self._history]

    @staticmethod
    def _timestamp() -> str:
        """Return a timezone-aware UTC timestamp in ISO 8601 format."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _emit_log(record: dict[str, Any]) -> None:
        """Emit one structured JSON log record."""
        logger.info(
            json.dumps(record, separators=(",", ":"), sort_keys=True)
        )


class StateLogger:
    """State transition logger used by the orchestrator."""

    def __init__(self, initial_state: AgentState = AgentState.IDLE) -> None:
        self.state: AgentState = AgentState(initial_state)
        self.events: list[dict[str, Any]] = []

    def transition(self, next_state: AgentState | str, reason: str = "") -> AgentState:
        from_state = self.state
        if isinstance(next_state, str):
            try:
                next_state = AgentState(next_state)
            except ValueError:
                next_state = AgentState[next_state]

        self.state = next_state
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "from_state": from_state.value if hasattr(from_state, "value") else str(from_state),
            "event": reason,
            "to_state": self.state.value if hasattr(self.state, "value") else str(self.state),
            "valid": True,
        }
        self.events.append(record)
        logger.info(json.dumps(record, separators=(",", ":"), sort_keys=True))
        return self.state

    def history(self) -> list[dict[str, Any]]:
        return [r.copy() for r in self.events]

