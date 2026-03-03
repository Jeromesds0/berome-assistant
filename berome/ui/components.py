"""Reusable Rich UI components used throughout Berome."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text


def make_console(theme=None) -> Console:
    from berome.ui.theme import BEROME_THEME
    return Console(theme=theme or BEROME_THEME, highlight=False)


def user_panel(text: str) -> Panel:
    return Panel(
        Text(text, style="bold white"),
        title="[berome.user] You [/berome.user]",
        border_style="green",
        padding=(0, 1),
    )


def assistant_panel(content: str, model: str = "") -> Panel:
    subtitle = f"[berome.muted]{model}[/berome.muted]" if model else ""
    return Panel(
        Markdown(content),
        title="[berome.assistant] Berome [/berome.assistant]",
        subtitle=subtitle,
        border_style="blue",
        padding=(0, 1),
    )


def error_panel(message: str) -> Panel:
    return Panel(
        Text(message, style="bold red"),
        title="[berome.error] Error [/berome.error]",
        border_style="red",
        padding=(0, 1),
    )


def success_panel(message: str) -> Panel:
    return Panel(
        Text(message, style="bold green"),
        title="[berome.success] Done [/berome.success]",
        border_style="green",
        padding=(0, 1),
    )


def agent_status_table(tasks: list[Any]) -> Table:
    table = Table(
        title="Active & Recent Tasks",
        box=box.ROUNDED,
        border_style="magenta",
        show_lines=False,
    )
    table.add_column("ID", style="dim", width=10)
    table.add_column("Name", style="bold white")
    table.add_column("Type", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Created", style="dim")

    status_styles = {
        "pending": "[yellow]⏳ pending[/yellow]",
        "running": "[blue]⚙ running[/blue]",
        "completed": "[green]✓ done[/green]",
        "failed": "[red]✗ failed[/red]",
        "cancelled": "[dim]✗ cancelled[/dim]",
    }

    for t in tasks:
        table.add_row(
            t.id,
            t.name or t.description[:40],
            t.agent_type,
            status_styles.get(t.status.value, t.status.value),
            t.created_at.strftime("%H:%M:%S"),
        )
    return table


def repo_table(repos: list[dict]) -> Table:
    table = Table(
        title="GitHub Repositories",
        box=box.SIMPLE_HEAVY,
        border_style="green",
    )
    table.add_column("Name", style="bold cyan")
    table.add_column("Stars", justify="right", style="yellow")
    table.add_column("Forks", justify="right", style="blue")
    table.add_column("Private", justify="center")
    table.add_column("URL", style="dim underline")

    for r in repos:
        table.add_row(
            r["name"],
            str(r.get("stars", 0)),
            str(r.get("forks", 0)),
            "🔒" if r.get("private") else "🌐",
            r.get("url", ""),
        )
    return table


def make_spinner(description: str = "Thinking…") -> Progress:
    return Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        transient=True,
    )


def help_panel() -> Panel:
    help_text = """\
**Chat commands**
  /help            Show this help
  /clear           Clear conversation history
  /history         Show conversation history
  /exit  /quit     Exit Berome

**GitHub commands**
  /gh repos                          List your repositories
  /gh repo <name>                    Show repo details
  /gh create <name> [--private]      Create a new repo
  /gh read <repo> <path>             Read a file from a repo
  /gh ls <repo> [path]               List directory contents
  /gh write <repo> <path> <msg>      Write/update file (prompts for content)
  /gh clone <repo> <local-dir>       Clone a repo locally
  /gh push <local-dir> <msg>         Commit & push local changes
  /gh branch <repo> <branch>         Create a branch
  /gh pr <repo> <head> <title>       Open a pull request

**Agent commands**
  /agents          Show all background tasks
  /agent run <type> [json-payload]   Dispatch a task manually
  /agent types     List registered agent types

**Settings**
  /provider        Show active LLM provider
  /provider set anthropic|ollama     Switch provider (runtime)
"""
    return Panel(Markdown(help_text), title="[bold cyan] Berome Help [/bold cyan]", border_style="cyan")
