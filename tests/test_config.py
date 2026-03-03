"""Tests for configuration loading."""

import os
import pytest
from berome.config import LLMProvider, Settings


def test_default_provider_is_anthropic():
    s = Settings()
    assert s.provider == LLMProvider.anthropic


def test_provider_coercion_from_string(monkeypatch):
    monkeypatch.setenv("BEROME_PROVIDER", "OLLAMA")
    s = Settings()
    assert s.provider == LLMProvider.ollama


def test_active_model_anthropic(monkeypatch):
    monkeypatch.setenv("BEROME_PROVIDER", "anthropic")
    monkeypatch.setenv("BEROME_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    s = Settings()
    assert s.active_model() == "claude-haiku-4-5-20251001"


def test_active_model_ollama(monkeypatch):
    monkeypatch.setenv("BEROME_PROVIDER", "ollama")
    monkeypatch.setenv("BEROME_OLLAMA_MODEL", "mistral")
    s = Settings()
    assert s.active_model() == "mistral"
