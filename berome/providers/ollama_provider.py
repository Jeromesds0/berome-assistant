"""Ollama local LLM provider (drop-in swap for Anthropic)."""

from __future__ import annotations

import json
import uuid
from typing import AsyncIterator

import httpx

from berome.providers.base import LLMMessage, LLMProvider, LLMResponse, ToolCall


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

    def _build_messages_with_tools(
        self, messages: list[LLMMessage], system: str
    ) -> list[dict]:
        """Build OpenAI-compatible messages list for tool-use calls."""
        result: list[dict] = []
        sys_parts = [system] if system else []
        sys_parts += [m.content for m in messages if m.role == "system"]
        if sys_parts:
            result.append({"role": "system", "content": "\n".join(sys_parts)})

        for msg in messages:
            if msg.role == "system":
                continue
            elif msg.role == "tool":
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
            elif msg.role == "assistant" and msg.tool_calls:
                result.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })
            else:
                result.append({"role": msg.role, "content": msg.content})
        return result

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert neutral tool defs to OpenAI/Ollama function format.

        Strips ``required: []`` empty arrays — Ollama rejects them.
        """
        result = []
        for t in tools:
            params = dict(t["parameters"])
            if "required" in params and not params["required"]:
                del params["required"]
            result.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": params,
                },
            })
        return result

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

    async def chat_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[dict],
        system: str = "",
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a conversation with tools; parse tool_calls from response."""
        payload = {
            "model": self._model,
            "messages": self._build_messages_with_tools(messages, system),
            "tools": self._convert_tools(tools),
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{self._base_url}/api/chat", json=payload)
            if r.status_code >= 400:
                body = r.text
                raise RuntimeError(
                    f"Ollama {r.status_code} — {body}\n"
                    f"Ensure model {self._model!r} supports tool calling "
                    f"(e.g. llama3.1, llama3.2, qwen2.5, mistral-nemo)."
                )
            data = r.json()

        msg = data.get("message", {})
        text_content: str = msg.get("content", "") or ""
        tool_calls: list[ToolCall] = []

        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tc_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
            tool_calls.append(
                ToolCall(name=fn.get("name", ""), arguments=args, id=tc_id)
            )

        return LLMResponse(
            content=text_content,
            model=self._model,
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
        )
