# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable, with Discord support)
python3 -m venv .venv
.venv/bin/pip install -e ".[discord]"

# Run the CLI assistant
.venv/bin/berome
.venv/bin/berome --provider ollama --model qwen2.5:14b

# Run the Discord bot
.venv/bin/berome-discord

# Run all tests
.venv/bin/pytest tests/ -v

# Run a single test
.venv/bin/pytest tests/test_agents.py::test_orchestrator_runs_registered_agent -v

# Lint
.venv/bin/ruff check berome/
```

## Architecture

**Data flow (CLI):** `cli.py` → `BeromeSession` → `ChatAgent` → `LLMProvider`

**Data flow (Discord):** `discord_bot.py` → per-channel `BeromeSession` → `ChatAgent` → `LLMProvider`

**Provider abstraction** (`berome/providers/`): `LLMProvider` ABC defines `chat()` and `stream()`. `AnthropicProvider` and `OllamaProvider` implement it. `factory.get_provider()` reads `settings.provider`. Hot-swapping via `BeromeSession.switch_provider()` rebuilds the provider and all agents. Only `AnthropicProvider` implements `chat_with_tools()` currently; the Discord bot falls back to `chat_stream()` when the method is absent.

**Agent system** (`berome/agents/`): `AgentOrchestrator` holds a registry of `Agent` subclasses (`chat`, `code`, `research`, `github`). Tasks are `AgentTask` dataclasses dispatched by type string under an `asyncio.Semaphore(BEROME_MAX_AGENTS)`. `ChatAgent` is also called directly for streaming via `BeromeSession.agentic_stream()` and `BeromeSession.continue_agentic_stream()`.

**Agentic tool loop** (`berome/tools/`): `ChatAgent.stream_agentic_response()` runs a multi-turn tool-use loop when the provider supports `chat_with_tools`. Tool definitions live in `tools/definitions.py`; dispatch and implementations are in `tools/executor.py`; dangerous-command filtering in `tools/safety.py`.

**Discord bot** (`berome/discord_bot.py`): Uses `discord.Client` (not `commands.Bot`) to avoid double-handling. One `BeromeSession` per channel, keyed by `channel.id`, with a 1-hour TTL. A per-channel `asyncio.Lock` (in `_channel_locks`) serialises concurrent message handlers to prevent history corruption on rapid messages. On session creation, recent channel history and guild memories are seeded into the chat history. Images in messages (attachments and embeds) are downloaded, base64-encoded, and passed as multimodal content blocks. Shell commands require a `ConfirmView` (discord.ui buttons) before execution. Documents written by `write_file` during the agentic loop are auto-attached as Discord file attachments; `.md` files are also converted to `.html` via the `markdown` package.

**Guild persistence** (`berome/guild_data.py`): Guild memories (`/teach`, `/forget`) and active channels (`/activate`) are stored as JSON files in `~/.berome/guilds/` and `~/.berome/active_channels.json`.

**Config** (`berome/config.py`): Single `Settings` pydantic-settings singleton imported everywhere as `settings`. All env vars use the `BEROME_` prefix (except `ANTHROPIC_API_KEY` and `GITHUB_TOKEN`). `.env` is auto-loaded from the working directory.

## Key patterns

- Everything async — providers, agents, tool execution, and both frontends use `async/await`
- `LLMMessage(role, content, tool_call_id, tool_calls)` is the universal message format across providers
- `LLMResponse` carries `content`, `input_tokens`, `output_tokens`, `stop_reason`, and `tool_calls`
- `ToolResult(tool_name, tool_call_id, output, error)` is returned from every tool execution
- Tests use `pytest-asyncio` in `STRICT` mode — mark async tests with `@pytest.mark.asyncio`
- Provider tests mock the underlying SDK client directly (e.g. `provider._client.messages.create = AsyncMock(...)`)
- System prompts are Markdown files in `berome/prompts/`, loaded via `berome/prompts/__init__.py:load()`
