"""Anthropic (Claude) LLM provider."""

from __future__ import annotations

from typing import AsyncIterator

import anthropic

from berome.providers.base import LLMMessage, LLMProvider, LLMResponse


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
