"""
Berome session – wires together the LLM provider, agent orchestrator,
and GitHub integration into one stateful object used by the CLI.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from berome.agents.base import AgentTask
from berome.agents.chat_agent import ChatAgent, SYSTEM_PROMPT
from berome.agents.code_agent import CodeAgent
from berome.agents.github_agent import GitHubAgent
from berome.agents.orchestrator import AgentOrchestrator
from berome.agents.research_agent import ResearchAgent
from berome.config import Settings, settings
from berome.providers.base import LLMProvider, LLMResponse
from berome.providers.factory import get_provider

logger = logging.getLogger(__name__)


class BeromeSession:
    """Top-level session object: one per CLI invocation."""

    def __init__(self, cfg: Settings = settings, system_prompt: str = SYSTEM_PROMPT) -> None:
        self.cfg = cfg
        self._system_prompt = system_prompt
        self._provider: LLMProvider = get_provider()
        self._orchestrator = AgentOrchestrator()
        self._chat_agent = ChatAgent(self._provider, system_prompt=self._system_prompt)
        self._setup_agents()
        # Token tracking
        self._session_input_tokens: int = 0
        self._session_output_tokens: int = 0
        self._last_input_tokens: int = 0
        self._last_output_tokens: int = 0

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
        from berome.config import Settings
        new_cfg = Settings()
        self.cfg = new_cfg
        self._provider = get_provider()
        # Rebuild agents with new provider, preserving any custom system prompt
        self._chat_agent = ChatAgent(self._provider, system_prompt=self._system_prompt)
        self._orchestrator = AgentOrchestrator()
        self._setup_agents()

    # ── Token tracking ────────────────────────────────────────────────────────

    def _accumulate_tokens(self, response: LLMResponse) -> None:
        """Record token usage from one LLM response."""
        self._last_input_tokens = response.input_tokens
        self._last_output_tokens = response.output_tokens
        self._session_input_tokens += response.input_tokens
        self._session_output_tokens += response.output_tokens

    def token_stats(self) -> dict:
        """Return current token usage counters."""
        return {
            "session_in": self._session_input_tokens,
            "session_out": self._session_output_tokens,
            "last_in": self._last_input_tokens,
            "last_out": self._last_output_tokens,
        }

    # ── Chat ──────────────────────────────────────────────────────────────────

    async def chat_stream(self, user_input: str):
        """Yield token chunks from the chat agent (simple streaming path)."""
        async for chunk in self._chat_agent.stream_response(user_input):
            yield chunk

    async def agentic_stream(
        self,
        user_input: str,
        on_tool_call: Callable[[str, dict], Awaitable[None]],
        on_tool_result: Callable,
        require_confirmation: Optional[Callable[[str], Awaitable[bool]]] = None,
    ):
        """
        Run the agentic tool-use loop, yielding the final text response.

        Falls back to plain streaming if the provider lacks ``chat_with_tools``.
        """
        from berome.tools.definitions import TOOL_DEFINITIONS

        if not hasattr(self._provider, "chat_with_tools"):
            async for chunk in self.chat_stream(user_input):
                yield chunk
            return

        # Reset per-response counters before starting
        self._last_input_tokens = 0
        self._last_output_tokens = 0

        async for chunk in self._chat_agent.stream_agentic_response(
            user_input=user_input,
            tools=TOOL_DEFINITIONS,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            require_confirmation=require_confirmation,
            on_llm_response=self._accumulate_tokens,
        ):
            yield chunk

    def clear_history(self) -> None:
        self._chat_agent.clear_history()

    def add_history_message(self, role: str, content: str) -> None:
        """Inject a message into the conversation history (e.g. for seeding context)."""
        self._chat_agent.add_message(role, content)

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
