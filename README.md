# Berome

A CLI-based AI personal assistant powered by **Ollama** (local) or **Anthropic Claude**, with GitHub integration and a background agent system.

```
 ██████╗ ███████╗██████╗  ██████╗ ███╗   ███╗███████╗
 ██╔══██╗██╔════╝██╔══██╗██╔═══██╗████╗ ████║██╔════╝
 ██████╔╝█████╗  ██████╔╝██║   ██║██╔████╔██║█████╗
 ██╔══██╗██╔══╝  ██╔══██╗██║   ██║██║╚██╔╝██║██╔══╝
 ██████╔╝███████╗██║  ██║╚██████╔╝██║ ╚═╝ ██║███████╗
 ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝
```

## Features

- **Streaming chat** — token-by-token responses rendered in a Rich terminal UI
- **Swappable LLM backends** — use Ollama locally or Anthropic Claude, switchable at runtime with `/provider set ollama`
- **GitHub integration** — create repos, read/write files, clone, commit & push, open PRs — all from the CLI
- **Background agent system** — async task orchestrator with specialised sub-agents (chat, code, research, GitHub)
- **Slash commands** — `/gh`, `/agents`, `/agent run`, `/clear`, `/history`, `/help` and more

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally (for local inference), or an Anthropic API key
- A GitHub personal access token (for GitHub commands)

## Installation

```bash
git clone https://github.com/Jeromesds0/berome-assistant.git
cd berome-assistant
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Configuration

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

```env
# Choose your backend: "ollama" or "anthropic"
BEROME_PROVIDER=ollama

# Ollama (local)
BEROME_OLLAMA_BASE_URL=http://localhost:11434
BEROME_OLLAMA_MODEL=qwen2.5:14b

# Anthropic (optional)
ANTHROPIC_API_KEY=sk-ant-...
BEROME_ANTHROPIC_MODEL=claude-sonnet-4-6

# GitHub (optional, needed for /gh commands)
GITHUB_TOKEN=ghp_...
GITHUB_USERNAME=your-username
```

> `.env` is gitignored — your keys will never be committed.

## Usage

```bash
.venv/bin/berome
```

Or with flags:

```bash
.venv/bin/berome --provider ollama --model qwen2.5:14b
.venv/bin/berome --provider anthropic
```

## Slash Commands

### Chat
| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/clear` | Clear conversation history |
| `/history` | Replay conversation so far |
| `/exit` | Quit Berome |

### GitHub
| Command | Description |
|---|---|
| `/gh repos` | List your repositories |
| `/gh repo <name>` | Show repo details |
| `/gh create <name> [--private]` | Create a new repo |
| `/gh read <repo> <path>` | Read a file from a repo |
| `/gh ls <repo> [path]` | List directory contents |
| `/gh write <repo> <path> <msg>` | Create or update a file |
| `/gh clone <repo> <local-dir>` | Clone a repo locally |
| `/gh push <local-dir> <msg>` | Commit & push local changes |
| `/gh branch <repo> <branch>` | Create a branch |
| `/gh pr <repo> <head> <title>` | Open a pull request |

### Agents
| Command | Description |
|---|---|
| `/agents` | Show all background task statuses |
| `/agent types` | List registered agent types |
| `/agent run <type> <json>` | Dispatch a task manually |
| `/provider` | Show active provider and model |
| `/provider set anthropic\|ollama` | Hot-swap LLM backend |

## Agent System

Berome runs tasks through an async orchestrator with four built-in agents:

| Agent | Type string | What it does |
|---|---|---|
| Chat | `chat` | Multi-turn conversation with streaming |
| Code | `code` | Write, review, explain, refactor code |
| Research | `research` | Topic summaries with configurable depth |
| GitHub | `github` | All GitHub operations as background tasks |

Dispatch a task manually:

```
/agent run research {"topic": "Rust async runtimes", "depth": "deep"}
/agent run code {"mode": "write", "prompt": "binary search in Python"}
```

## Swapping to Anthropic

```bash
# In .env
BEROME_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Or at runtime inside Berome
/provider set anthropic
```

## Recommended Ollama Models

| Model | VRAM | Best for |
|---|---|---|
| `qwen2.5:14b` | ~9GB | Best general quality (recommended) |
| `qwen2.5-coder:14b` | ~9GB | Coding tasks |
| `deepseek-r1:14b` | ~9GB | Reasoning / thinking |
| `llama3.1:8b` | ~5GB | Fast, lower VRAM |
| `llama3.2:3b` | ~2GB | Instant responses |

## Project Structure

```
berome/
├── config.py              # pydantic-settings config
├── session.py             # top-level session (wires everything together)
├── cli.py                 # Typer CLI + interactive loop
├── providers/             # LLM backend abstraction
│   ├── base.py            #   LLMProvider ABC
│   ├── anthropic_provider.py
│   ├── ollama_provider.py
│   └── factory.py         #   provider factory
├── agents/                # Agent system
│   ├── base.py            #   Agent ABC + AgentTask
│   ├── orchestrator.py    #   async task pool
│   ├── chat_agent.py
│   ├── code_agent.py
│   ├── github_agent.py
│   └── research_agent.py
├── integrations/
│   └── github.py          # PyGithub + git CLI wrapper
└── ui/
    ├── theme.py            # Rich theme + banner
    └── components.py       # panels, tables, spinners
```

## Running Tests

```bash
.venv/bin/pytest tests/ -v
```

## License

MIT
