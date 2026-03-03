"""Basic unit tests for the provider abstraction (no real API calls)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from berome.providers.base import LLMMessage, LLMResponse
from berome.providers.anthropic_provider import AnthropicProvider
from berome.providers.ollama_provider import OllamaProvider


# ── AnthropicProvider ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_anthropic_chat_returns_response():
    provider = AnthropicProvider(api_key="test-key", model="claude-haiku-4-5-20251001")

    # Patch the underlying SDK client
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello!")]
    mock_response.model = "claude-haiku-4-5-20251001"
    mock_response.usage.input_tokens = 5
    mock_response.usage.output_tokens = 3
    mock_response.stop_reason = "end_turn"

    provider._client.messages.create = AsyncMock(return_value=mock_response)

    result = await provider.chat([LLMMessage(role="user", content="hi")])
    assert result.content == "Hello!"
    assert result.model == "claude-haiku-4-5-20251001"


def test_anthropic_provider_name():
    p = AnthropicProvider(api_key="x")
    assert p.provider_name == "anthropic"


# ── OllamaProvider ────────────────────────────────────────────────────────────


def test_ollama_provider_name():
    p = OllamaProvider(model="llama3.1")
    assert p.provider_name == "ollama"
    assert p.model_name == "llama3.1"


def test_ollama_build_payload_includes_system():
    p = OllamaProvider()
    payload = p._build_payload(
        [LLMMessage(role="user", content="hello")],
        system="Be helpful",
        max_tokens=100,
        temperature=0.5,
        stream=False,
    )
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"] == "Be helpful"
    assert payload["messages"][1]["content"] == "hello"
    assert payload["stream"] is False
