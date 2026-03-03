"""Abstract base class that every LLM backend must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal


@dataclass
class LLMMessage:
    role: Literal["user", "assistant", "system"]
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = "end_turn"


class LLMProvider(ABC):
    """Uniform interface for any LLM backend."""

    # ── Required ──────────────────────────────────────────────────────────────

    @abstractmethod
    async def chat(
        self,
        messages: list[LLMMessage],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Send a conversation and return the full response."""

    @abstractmethod
    async def stream(
        self,
        messages: list[LLMMessage],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Yield response text chunks as they arrive."""

    # ── Optional helpers ──────────────────────────────────────────────────────

    async def ping(self) -> bool:
        """Return True if the provider is reachable."""
        try:
            resp = await self.chat(
                [LLMMessage(role="user", content="ping")],
                max_tokens=10,
            )
            return bool(resp.content)
        except Exception:
            return False

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...
