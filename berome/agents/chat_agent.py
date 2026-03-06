"""Chat agent – drives a multi-turn conversation with the LLM."""

from __future__ import annotations

from typing import AsyncIterator, Awaitable, Callable, Optional

from berome.agents.base import Agent, AgentTask
from berome.prompts import load as _load_prompt
from berome.providers.base import LLMMessage, LLMProvider, LLMResponse


SYSTEM_PROMPT = _load_prompt("cli_system.md")


class ChatAgent(Agent):
    agent_type = "chat"
    description = "Handles multi-turn conversation with the LLM"

    def __init__(self, provider: LLMProvider, system_prompt: str = SYSTEM_PROMPT) -> None:
        super().__init__(provider)
        self._history: list[LLMMessage] = []
        self._system_prompt = system_prompt

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
            system=self._system_prompt,
        )
        self.add_message("assistant", response.content)
        return response.content

    async def stream_response(self, user_input: str) -> AsyncIterator[str]:
        """Stream tokens for the given input, updating history when done."""
        self.add_message("user", user_input)
        full_response = []
        async for chunk in self._llm.stream(
            messages=self._history,
            system=self._system_prompt,
        ):
            full_response.append(chunk)
            yield chunk
        self.add_message("assistant", "".join(full_response))

    async def stream_agentic_response(
        self,
        user_input: str,
        tools: list[dict],
        on_tool_call: Callable[[str, dict], Awaitable[None]],
        on_tool_result: Callable,
        require_confirmation: Optional[Callable[[str], Awaitable[bool]]] = None,
        on_llm_response: Optional[Callable[[LLMResponse], None]] = None,
    ) -> AsyncIterator[str]:
        """
        Agentic tool-use loop.

        Calls ``chat_with_tools`` on the provider, executes any tool calls,
        feeds results back, and repeats until the LLM returns plain text
        (or max 20 iterations).  The final text response is yielded as a
        single chunk.
        """
        from berome.tools.executor import execute_tool

        self.add_message("user", user_input)

        for _iteration in range(20):
            response = await self._llm.chat_with_tools(  # type: ignore[attr-defined]
                messages=self._history,
                tools=tools,
                system=self._system_prompt,
            )

            if on_llm_response is not None:
                on_llm_response(response)

            if not response.tool_calls:
                # Final text — yield and finish
                self.add_message("assistant", response.content)
                yield response.content
                return

            # Assistant turn with tool calls — store in history
            self._history.append(
                LLMMessage(
                    role="assistant",
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                )
            )

            # Execute each tool and accumulate results
            for tc in response.tool_calls:
                await on_tool_call(tc.name, tc.arguments)
                result = await execute_tool(
                    tc.name, tc.arguments, tc.id, require_confirmation
                )
                await on_tool_result(result)
                # Store tool result in history
                self._history.append(
                    LLMMessage(
                        role="tool",
                        content=result.output,
                        tool_call_id=tc.id,
                    )
                )

        # Reached iteration limit
        self.add_message("assistant", "Maximum tool use iterations reached.")
        yield "Maximum tool use iterations reached."
