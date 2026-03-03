"""Factory – return the correct LLMProvider from settings."""

from __future__ import annotations

from berome.config import LLMProvider as ProviderEnum, settings
from berome.providers.base import LLMProvider


def get_provider() -> LLMProvider:
    """Instantiate and return the configured LLM provider."""
    if settings.provider == ProviderEnum.anthropic:
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file or environment."
            )
        from berome.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
        )

    if settings.provider == ProviderEnum.ollama:
        from berome.providers.ollama_provider import OllamaProvider

        return OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )

    raise ValueError(f"Unknown provider: {settings.provider!r}")
