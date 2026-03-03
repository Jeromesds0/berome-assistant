"""
Berome session – wires together the LLM provider, agent orchestrator,
and GitHub integration into one stateful object used by the CLI.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from berome.agents.base import AgentTask
from berome.agents.chat_agent import ChatAgent
from berome.agents.code_agent import CodeAgent
from berome.agents.github_agent import GitHubAgent
from berome.agents.orchestrator import AgentOrchestrator
from berome.agents.research_agent import ResearchAgent
from berome.config import Settings, settings
from berome.providers.base import LLMProvider
from berome.providers.factory import get_provider

logger = logging.getLogger(__name__)


class BeromeSession:
    """Top-level session object: one per CLI invocation."""

    def __init__(self, cfg: Settings = settings) -> None:
        self.cfg = cfg
        self._provider: LLMProvider = get_provider()
        self._orchestrator = AgentOrchestrator()
        self._chat_agent = ChatAgent(self._provider)
        self._setup_agents()

    def _setup_agents(self) -> None:
        self._orchestrator.register(self._chat_agent)
        self._orchestrator.register(CodeAgent(self._provider))
        self._orchestrator.register(ResearchAgent(self._provider))
        self._orchestrator.register(GitHubAgent())

    # ── Provider ──────────────────────────────────────────────────────────────

    @property
    def provider(self) -> LLMProvider:
        return self._provider

    def switch_provider(self, provider_name: str) -> None:
        """Hot-swap the LLM provider at runtime."""
        import os
        os.environ["BEROME_PROVIDER"] = provider_name
        # Reload settings and rebuild provider
        from berome.config import LLMProvider as PE, Settings
        new_cfg = Settings()
        self.cfg = new_cfg
        self._provider = get_provider()
        # Rebuild agents with new provider
        self._chat_agent = ChatAgent(self._provider)
        self._orchestrator = AgentOrchestrator()
        self._setup_agents()

    # ── Chat ──────────────────────────────────────────────────────────────────

    async def chat_stream(self, user_input: str):
        """Yield token chunks from the chat agent."""
        async for chunk in self._chat_agent.stream_response(user_input):
            yield chunk

    def clear_history(self) -> None:
        self._chat_agent.clear_history()

    def history(self):
        return self._chat_agent.history

    # ── Background tasks ──────────────────────────────────────────────────────

    def dispatch_task(self, task: AgentTask, on_complete=None) -> AgentTask:
        return self._orchestrator.dispatch(task, on_complete)

    async def run_task(self, task: AgentTask) -> AgentTask:
        return await self._orchestrator.run_task(task)

    def agent_summary(self) -> dict:
        return self._orchestrator.summary()

    def all_tasks(self):
        return self._orchestrator.all_tasks()

    def registered_agent_types(self) -> list[str]:
        return self._orchestrator.registered_agents()

    # ── GitHub (direct, blocking) ─────────────────────────────────────────────

    def github(self):
        """Return a live GitHubIntegration (raises if token not set)."""
        from berome.integrations.github import GitHubIntegration
        return GitHubIntegration()
