"""Chat agent – drives a multi-turn conversation with the LLM."""

from __future__ import annotations

from typing import AsyncIterator

from berome.agents.base import Agent, AgentTask
from berome.providers.base import LLMMessage, LLMProvider


SYSTEM_PROMPT = """\
You are Berome, a highly capable AI personal assistant.
You can help with coding, research, project management, and GitHub operations.
Be concise, precise, and always proactively suggest next steps when relevant.
When the user asks you to perform an action (GitHub, files, etc.), clearly
state what you're going to do and confirm the result.\
"""


class ChatAgent(Agent):
    agent_type = "chat"
    description = "Handles multi-turn conversation with the LLM"

    def __init__(self, provider: LLMProvider) -> None:
        super().__init__(provider)
        self._history: list[LLMMessage] = []

    def add_message(self, role: str, content: str) -> None:
        self._history.append(LLMMessage(role=role, content=content))  # type: ignore[arg-type]

    def clear_history(self) -> None:
        self._history.clear()

    @property
    def history(self) -> list[LLMMessage]:
        return list(self._history)

    async def run(self, task: AgentTask) -> str:
        user_input: str = task.payload.get("input", "")
        self.add_message("user", user_input)
        response = await self._llm.chat(
            messages=self._history,
            system=SYSTEM_PROMPT,
        )
        self.add_message("assistant", response.content)
        return response.content

    async def stream_response(self, user_input: str) -> AsyncIterator[str]:
        """Stream tokens for the given input, updating history when done."""
        self.add_message("user", user_input)
        full_response = []
        async for chunk in self._llm.stream(
            messages=self._history,
            system=SYSTEM_PROMPT,
        ):
            full_response.append(chunk)
            yield chunk
        self.add_message("assistant", "".join(full_response))
