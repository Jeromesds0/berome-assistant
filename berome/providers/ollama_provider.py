"""Ollama local LLM provider (drop-in swap for Anthropic)."""

from __future__ import annotations

from typing import AsyncIterator

import httpx

from berome.providers.base import LLMMessage, LLMProvider, LLMResponse


class OllamaProvider(LLMProvider):
    """Calls the Ollama REST API at http://localhost:11434 (or configured URL)."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1") -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    def _build_payload(
        self,
        messages: list[LLMMessage],
        system: str,
        max_tokens: int,
        temperature: float,
        stream: bool,
    ) -> dict:
        ollama_messages: list[dict] = []
        sys_parts = [system] if system else []
        sys_parts += [m.content for m in messages if m.role == "system"]
        if sys_parts:
            ollama_messages.append({"role": "system", "content": "\n".join(sys_parts)})
        ollama_messages += [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role != "system"
        ]
        return {
            "model": self._model,
            "messages": ollama_messages,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

    async def chat(
        self,
        messages: list[LLMMessage],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        payload = self._build_payload(messages, system, max_tokens, temperature, stream=False)
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{self._base_url}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
        return LLMResponse(
            content=data["message"]["content"],
            model=self._model,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        import json

        payload = self._build_payload(messages, system, max_tokens, temperature, stream=True)
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", f"{self._base_url}/api/chat", json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        if text := chunk.get("message", {}).get("content"):
                            yield text
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
