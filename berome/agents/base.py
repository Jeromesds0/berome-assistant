"""Base agent types shared by all sub-agents."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional


class AgentStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


@dataclass
class AgentTask:
    """A unit of work dispatched to a sub-agent."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    agent_type: str = "generic"
    payload: dict[str, Any] = field(default_factory=dict)
    status: AgentStatus = AgentStatus.pending
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    on_complete: Optional[Callable[["AgentTask"], None]] = None

    def mark_complete(self, result: Any) -> None:
        self.status = AgentStatus.completed
        self.result = result
        self.completed_at = datetime.now(timezone.utc)
        if self.on_complete:
            self.on_complete(self)

    def mark_failed(self, error: str) -> None:
        self.status = AgentStatus.failed
        self.error = error
        self.completed_at = datetime.now(timezone.utc)


class Agent:
    """Base class every sub-agent must extend."""

    agent_type: str = "generic"
    description: str = "Generic agent"

    def __init__(self, llm_provider=None) -> None:
        self._llm = llm_provider

    async def run(self, task: AgentTask) -> Any:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<Agent type={self.agent_type!r}>"
