# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in editable mode
python3 -m venv .venv && .venv/bin/pip install -e .

# Run the assistant
.venv/bin/berome
.venv/bin/berome --provider ollama --model qwen2.5:14b
.venv/bin/berome --provider anthropic

# Run all tests
.venv/bin/pytest tests/ -v

# Run a single test
.venv/bin/pytest tests/test_agents.py::test_orchestrator_runs_registered_agent -v

# Lint
.venv/bin/ruff check berome/
```

## Architecture

**Data flow:** `cli.py` → `BeromeSession` → `ChatAgent` → `LLMProvider`

**Provider abstraction** (`berome/providers/`): `LLMProvider` ABC defines `chat()` and `stream()`. `AnthropicProvider` and `OllamaProvider` implement it. `factory.get_provider()` reads `settings.provider` to instantiate the right one. Hot-swapping at runtime is done via `BeromeSession.switch_provider()`, which rebuilds the provider and all agents.

**Agent system** (`berome/agents/`): `AgentOrchestrator` holds a registry of `Agent` subclasses (chat, code, research, github). Tasks are `AgentTask` dataclasses dispatched by type string. The orchestrator runs tasks under an asyncio semaphore (`BEROME_MAX_AGENTS`). `ChatAgent` is special — it is also called directly for streaming responses via `BeromeSession.agentic_stream()`.

**Agentic tool loop** (`berome/tools/`): When the provider supports `chat_with_tools`, `ChatAgent.stream_agentic_response()` runs a multi-turn tool-use loop. Tool definitions live in `tools/definitions.py`; execution (with safety checks) is in `tools/executor.py` and `tools/safety.py`.

**Config** (`berome/config.py`): Single `Settings` pydantic-settings singleton imported as `settings`. All env vars use the `BEROME_` prefix (except `ANTHROPIC_API_KEY` and `GITHUB_TOKEN`). The `.env` file is auto-loaded.

**CLI** (`berome/cli.py`): Typer entry-point → `asyncio.run(_chat_loop())`. Slash commands are handled in `_handle_command()`. GitHub sub-commands in `_handle_gh_command()`, agent sub-commands in `_handle_agent_command()`.

## Key patterns

- Everything async — providers, agents, and the chat loop all use `async/await`
- `LLMMessage` dataclass (`role`, `content`, `tool_call_id`, `tool_calls`) is the universal message format passed to providers
- `LLMResponse` carries `content`, `input_tokens`, `output_tokens`, `stop_reason`, and `tool_calls`
- Tests use `pytest-asyncio` in `STRICT` mode — mark async tests with `@pytest.mark.asyncio`
- Provider tests mock the underlying SDK client directly (e.g., `provider._client.messages.create = AsyncMock(...)`)
