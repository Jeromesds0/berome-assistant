"""Tests for the agent system."""

import asyncio
import pytest

from berome.agents.base import AgentTask, AgentStatus
from berome.agents.orchestrator import AgentOrchestrator
from berome.agents.base import Agent


class EchoAgent(Agent):
    agent_type = "echo"

    async def run(self, task: AgentTask):
        return f"echo: {task.payload.get('msg', '')}"


@pytest.mark.asyncio
async def test_orchestrator_runs_registered_agent():
    orch = AgentOrchestrator()
    orch.register(EchoAgent())

    task = AgentTask(name="test", agent_type="echo", payload={"msg": "hello"})
    result_task = await orch.run_task(task)

    assert result_task.status == AgentStatus.completed
    assert result_task.result == "echo: hello"


@pytest.mark.asyncio
async def test_orchestrator_fails_unknown_agent():
    orch = AgentOrchestrator()
    task = AgentTask(name="test", agent_type="nonexistent", payload={})
    result_task = await orch.run_task(task)

    assert result_task.status == AgentStatus.failed
    assert "No agent registered" in result_task.error


@pytest.mark.asyncio
async def test_orchestrator_summary():
    orch = AgentOrchestrator()
    orch.register(EchoAgent())

    tasks = [
        AgentTask(agent_type="echo", payload={"msg": str(i)}) for i in range(3)
    ]
    for t in tasks:
        await orch.run_task(t)

    summary = orch.summary()
    assert summary["total"] == 3
    assert summary["completed"] == 3
    assert summary["failed"] == 0
