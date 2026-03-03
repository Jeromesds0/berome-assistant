"""Anthropic (Claude) LLM provider."""

from __future__ import annotations

from typing import AsyncIterator

import anthropic

from berome.providers.base import LLMMessage, LLMProvider, LLMResponse, ToolCall


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6") -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    def _build_messages(self, messages: list[LLMMessage]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]

    def _build_messages_with_tools(self, messages: list[LLMMessage]) -> list[dict]:
        """Build message list that handles tool_call / tool result turns."""
        result: list[dict] = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.role == "system":
                i += 1
                continue
            elif msg.role == "assistant" and msg.tool_calls:
                # Assistant turn that contains tool_use blocks
                content_blocks: list[dict] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                result.append({"role": "assistant", "content": content_blocks})
                i += 1
            elif msg.role == "tool":
                # Collect consecutive tool results into one user message
                tool_results: list[dict] = []
                while i < len(messages) and messages[i].role == "tool":
                    m = messages[i]
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": m.content,
                    })
                    i += 1
                result.append({"role": "user", "content": tool_results})
            else:
                result.append({"role": msg.role, "content": msg.content})
                i += 1
        return result

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert neutral tool defs (with 'parameters') to Anthropic format."""
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ]

    async def chat(
        self,
        messages: list[LLMMessage],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        # Collect system prompt from system arg or system-role messages
        sys_parts = [system] if system else []
        sys_parts += [m.content for m in messages if m.role == "system"]
        sys_text = "\n".join(sys_parts).strip()

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=sys_text or anthropic.NOT_GIVEN,
            messages=self._build_messages(messages),
        )
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason or "end_turn",
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        sys_parts = [system] if system else []
        sys_parts += [m.content for m in messages if m.role == "system"]
        sys_text = "\n".join(sys_parts).strip()

        async with self._client.messages.stream(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=sys_text or anthropic.NOT_GIVEN,
            messages=self._build_messages(messages),
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def chat_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[dict],
        system: str = "",
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a conversation with tool definitions; parse tool_use responses."""
        sys_parts = [system] if system else []
        sys_parts += [m.content for m in messages if m.role == "system"]
        sys_text = "\n".join(sys_parts).strip()

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=sys_text or anthropic.NOT_GIVEN,
            tools=self._convert_tools(tools),
            messages=self._build_messages_with_tools(messages),
        )

        # Parse content blocks
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(name=block.name, arguments=block.input, id=block.id)
                )

        return LLMResponse(
            content="".join(text_parts),
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason or "end_turn",
            tool_calls=tool_calls,
        )
