"""
Agent Orchestrator – manages a pool of sub-agents, dispatches tasks,
and runs them concurrently in the background via asyncio.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Callable, Optional

from berome.agents.base import Agent, AgentStatus, AgentTask
from berome.config import settings

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Central hub that:
    - Registers available sub-agents
    - Accepts tasks and dispatches to the right agent
    - Runs tasks in async background
    - Maintains a task history
    """

    def __init__(self) -> None:
        self._registry: dict[str, Agent] = {}
        self._tasks: dict[str, AgentTask] = {}
        self._semaphore = asyncio.Semaphore(settings.max_agents)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, agent: Agent) -> None:
        """Register a sub-agent by its type string."""
        self._registry[agent.agent_type] = agent
        logger.debug("Registered agent: %s", agent.agent_type)

    def registered_agents(self) -> list[str]:
        return list(self._registry.keys())

    # ── Task dispatch ─────────────────────────────────────────────────────────

    def dispatch(
        self,
        task: AgentTask,
        on_complete: Optional[Callable[[AgentTask], None]] = None,
    ) -> AgentTask:
        """
        Queue a task for execution. Returns the task immediately.
        The task runs in the background; attach on_complete for a callback.
        """
        if on_complete:
            task.on_complete = on_complete
        self._tasks[task.id] = task

        # Schedule on the running loop (works inside an async context)
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self._run_task(task))
        except RuntimeError:
            # No running loop – caller must await run_task directly
            pass

        return task

    async def run_task(self, task: AgentTask) -> AgentTask:
        """Await a task directly (useful when already in async context)."""
        self._tasks[task.id] = task
        await self._run_task(task)
        return task

    async def _run_task(self, task: AgentTask) -> None:
        agent = self._registry.get(task.agent_type)
        if not agent:
            task.mark_failed(f"No agent registered for type '{task.agent_type}'")
            logger.warning("Unknown agent type: %s", task.agent_type)
            return

        async with self._semaphore:
            task.status = AgentStatus.running
            logger.info("Agent[%s] starting task %s", task.agent_type, task.id)
            try:
                result = await asyncio.wait_for(
                    agent.run(task), timeout=settings.agent_timeout
                )
                task.mark_complete(result)
                logger.info("Agent[%s] task %s completed", task.agent_type, task.id)
            except asyncio.TimeoutError:
                task.mark_failed(f"Timed out after {settings.agent_timeout}s")
                logger.error("Task %s timed out", task.id)
            except Exception as exc:
                task.mark_failed(str(exc))
                logger.exception("Task %s failed: %s", task.id, exc)

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_task(self, task_id: str) -> Optional[AgentTask]:
        return self._tasks.get(task_id)

    def all_tasks(self) -> list[AgentTask]:
        return list(self._tasks.values())

    def pending_tasks(self) -> list[AgentTask]:
        return [t for t in self._tasks.values() if t.status == AgentStatus.pending]

    def running_tasks(self) -> list[AgentTask]:
        return [t for t in self._tasks.values() if t.status == AgentStatus.running]

    def completed_tasks(self) -> list[AgentTask]:
        return [t for t in self._tasks.values() if t.status == AgentStatus.completed]

    def failed_tasks(self) -> list[AgentTask]:
        return [t for t in self._tasks.values() if t.status == AgentStatus.failed]

    def summary(self) -> dict:
        tasks = self.all_tasks()
        return {
            "total": len(tasks),
            "pending": sum(1 for t in tasks if t.status == AgentStatus.pending),
            "running": sum(1 for t in tasks if t.status == AgentStatus.running),
            "completed": sum(1 for t in tasks if t.status == AgentStatus.completed),
            "failed": sum(1 for t in tasks if t.status == AgentStatus.failed),
        }
