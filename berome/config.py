"""Central configuration for Berome, loaded from environment / .env file."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    anthropic = "anthropic"
    ollama = "ollama"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="BEROME_",
        extra="ignore",
    )

    # ── Provider selection ────────────────────────────────────────────────────
    provider: LLMProvider = Field(LLMProvider.anthropic, alias="BEROME_PROVIDER")

    # ── Anthropic ─────────────────────────────────────────────────────────────
    anthropic_api_key: Optional[str] = Field(None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field("claude-sonnet-4-6", alias="BEROME_ANTHROPIC_MODEL")

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = Field("http://localhost:11434", alias="BEROME_OLLAMA_BASE_URL")
    ollama_model: str = Field("llama3.1", alias="BEROME_OLLAMA_MODEL")

    # ── GitHub ────────────────────────────────────────────────────────────────
    github_token: Optional[str] = Field(None, alias="GITHUB_TOKEN")
    github_username: Optional[str] = Field(None, alias="GITHUB_USERNAME")

    # ── Agent system ──────────────────────────────────────────────────────────
    max_agents: int = Field(5, alias="BEROME_MAX_AGENTS")
    agent_timeout: int = Field(120, alias="BEROME_AGENT_TIMEOUT")  # seconds

    # ── History / storage ─────────────────────────────────────────────────────
    history_dir: Path = Field(
        default_factory=lambda: Path.home() / ".berome" / "history"
    )

    # ── Discord ────────────────────────────────────────────────────────────────
    discord_token: Optional[str] = Field(None, alias="DISCORD_BOT_TOKEN")
    discord_require_mention: bool = Field(True, alias="DISCORD_REQUIRE_MENTION")
    # Comma-separated channel IDs where bot responds without @mention
    discord_allowed_channels: str = Field("", alias="DISCORD_ALLOWED_CHANNELS")
    # Auto-approve shell command confirmations (only for trusted/private servers)
    discord_auto_approve_tools: bool = Field(False, alias="DISCORD_AUTO_APPROVE_TOOLS")

    @field_validator("provider", mode="before")
    @classmethod
    def _coerce_provider(cls, v: str) -> str:
        return v.lower() if isinstance(v, str) else v

    def active_model(self) -> str:
        """Return the model identifier for the currently active provider."""
        return (
            self.anthropic_model
            if self.provider == LLMProvider.anthropic
            else self.ollama_model
        )

    def discord_allowed_channel_ids(self) -> set[int]:
        """Parse DISCORD_ALLOWED_CHANNELS into a set of channel ID ints."""
        if not self.discord_allowed_channels:
            return set()
        return {int(c.strip()) for c in self.discord_allowed_channels.split(",") if c.strip()}


# Module-level singleton – import this everywhere
settings = Settings()
