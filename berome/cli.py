"""
Berome CLI entry-point.

Usage:
  berome          – start interactive chat session
  berome --help   – show options
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Optional

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from berome.agents.base import AgentTask
from berome.config import settings
from berome.ui.components import (
    agent_status_table,
    assistant_panel,
    error_panel,
    help_panel,
    make_console,
    repo_table,
    success_panel,
    user_panel,
)
from berome.ui.theme import BANNER, BEROME_THEME

app = typer.Typer(
    name="berome",
    help="Berome – your CLI AI personal assistant",
    add_completion=False,
)

logging.basicConfig(level=logging.WARNING)


# ── Main command ──────────────────────────────────────────────────────────────


@app.command()
def main(
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="anthropic | ollama"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override model name"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Start an interactive Berome session."""
    if debug:
        logging.basicConfig(level=logging.DEBUG, force=True)

    if provider:
        import os
        os.environ["BEROME_PROVIDER"] = provider
    if model:
        import os
        prov = (provider or settings.provider.value)
        os.environ[f"BEROME_{prov.upper()}_MODEL"] = model

    console = make_console()
    console.print(BANNER)

    try:
        from berome.session import BeromeSession
        session = BeromeSession()
    except RuntimeError as exc:
        console.print(error_panel(str(exc)))
        raise typer.Exit(1)

    console.print(
        f"[berome.muted]Provider:[/berome.muted] [bold]{session.provider.provider_name}[/bold]  "
        f"[berome.muted]Model:[/berome.muted] [bold]{session.provider.model_name}[/bold]\n"
    )

    asyncio.run(_chat_loop(session, console))


# ── Interactive loop ──────────────────────────────────────────────────────────


async def _chat_loop(session, console: Console) -> None:
    prompt_session: PromptSession = PromptSession(history=InMemoryHistory())

    while True:
        try:
            raw = await asyncio.get_event_loop().run_in_executor(
                None, lambda: prompt_session.prompt("\n[You] > ")
            )
        except (EOFError, KeyboardInterrupt):
            console.print("\n[berome.muted]Goodbye![/berome.muted]")
            break

        user_input = raw.strip()
        if not user_input:
            continue

        # ── Slash commands ────────────────────────────────────────────────────
        if user_input.startswith("/"):
            await _handle_command(user_input, session, console)
            continue

        # ── Regular chat ──────────────────────────────────────────────────────
        console.print(user_panel(user_input))
        await _stream_chat(user_input, session, console)


async def _stream_chat(user_input: str, session, console: Console) -> None:
    """Stream the assistant response with live updating panel."""
    buffer: list[str] = []

    panel = Panel(
        Text(""),
        title="[berome.assistant] Berome [/berome.assistant]",
        border_style="blue",
        padding=(0, 1),
    )

    with Live(panel, console=console, refresh_per_second=15) as live:
        try:
            async for chunk in session.chat_stream(user_input):
                buffer.append(chunk)
                current = "".join(buffer)
                panel = Panel(
                    Markdown(current),
                    title=f"[berome.assistant] Berome [/berome.assistant]",
                    subtitle=f"[berome.muted]{session.provider.model_name}[/berome.muted]",
                    border_style="blue",
                    padding=(0, 1),
                )
                live.update(panel)
        except Exception as exc:
            console.print(error_panel(f"Stream error: {exc}"))


# ── Slash command dispatcher ──────────────────────────────────────────────────


async def _handle_command(raw: str, session, console: Console) -> None:
    parts = raw.lstrip("/").split()
    cmd = parts[0].lower() if parts else ""
    args = parts[1:]

    # ── Meta ──────────────────────────────────────────────────────────────────
    if cmd in ("help", "h", "?"):
        console.print(help_panel())

    elif cmd in ("exit", "quit", "q"):
        console.print("[berome.muted]Goodbye![/berome.muted]")
        raise SystemExit(0)

    elif cmd == "clear":
        session.clear_history()
        console.print("[berome.success]Conversation history cleared.[/berome.success]")

    elif cmd == "history":
        for msg in session.history():
            style = "green" if msg.role == "user" else "blue"
            label = "You" if msg.role == "user" else "Berome"
            console.print(
                Panel(Markdown(msg.content), title=f"[{style}]{label}[/{style}]", border_style=style)
            )

    # ── Provider ──────────────────────────────────────────────────────────────
    elif cmd == "provider":
        if args and args[0] == "set" and len(args) >= 2:
            new_provider = args[1]
            try:
                session.switch_provider(new_provider)
                console.print(success_panel(f"Switched to provider: {new_provider}"))
            except Exception as exc:
                console.print(error_panel(str(exc)))
        else:
            console.print(
                f"[berome.muted]Active provider:[/berome.muted] "
                f"[bold]{session.provider.provider_name}[/bold] / "
                f"[bold]{session.provider.model_name}[/bold]"
            )

    # ── Agents ────────────────────────────────────────────────────────────────
    elif cmd == "agents":
        tasks = session.all_tasks()
        if tasks:
            console.print(agent_status_table(tasks))
        else:
            console.print("[berome.muted]No tasks dispatched yet.[/berome.muted]")

    elif cmd == "agent":
        await _handle_agent_command(args, session, console)

    # ── GitHub ────────────────────────────────────────────────────────────────
    elif cmd == "gh":
        await _handle_gh_command(args, session, console)

    else:
        console.print(error_panel(f"Unknown command: /{cmd}  — try /help"))


# ── Agent sub-commands ────────────────────────────────────────────────────────


async def _handle_agent_command(args: list[str], session, console: Console) -> None:
    if not args:
        console.print(help_panel())
        return

    sub = args[0].lower()

    if sub == "types":
        types = session.registered_agent_types()
        console.print(f"[berome.agent]Registered agents:[/berome.agent] {', '.join(types)}")

    elif sub == "run":
        if len(args) < 2:
            console.print(error_panel("Usage: /agent run <type> [json-payload]"))
            return
        agent_type = args[1]
        payload: dict = {}
        if len(args) >= 3:
            try:
                payload = json.loads(" ".join(args[2:]))
            except json.JSONDecodeError as exc:
                console.print(error_panel(f"Invalid JSON payload: {exc}"))
                return
        task = AgentTask(name=f"Manual/{agent_type}", agent_type=agent_type, payload=payload)
        console.print(f"[berome.muted]Dispatching task [bold]{task.id}[/bold]…[/berome.muted]")
        completed = await session.run_task(task)
        if completed.status.value == "completed":
            result_str = (
                json.dumps(completed.result, indent=2)
                if isinstance(completed.result, (dict, list))
                else str(completed.result)
            )
            console.print(assistant_panel(result_str, model=session.provider.model_name))
        else:
            console.print(error_panel(f"Task failed: {completed.error}"))
    else:
        console.print(error_panel(f"Unknown agent sub-command: {sub}"))


# ── GitHub sub-commands ───────────────────────────────────────────────────────


async def _handle_gh_command(args: list[str], session, console: Console) -> None:
    if not args:
        console.print(help_panel())
        return

    sub = args[0].lower()

    try:
        gh = session.github()
    except RuntimeError as exc:
        console.print(error_panel(str(exc)))
        return

    # list repos
    if sub == "repos":
        repos = gh.list_repos()
        console.print(repo_table([r.__dict__ for r in repos]))

    # show repo
    elif sub == "repo":
        if not args[1:]:
            console.print(error_panel("Usage: /gh repo <name>"))
            return
        info = gh.get_repo(args[1])
        _print_repo_detail(info, console)

    # create repo
    elif sub == "create":
        if not args[1:]:
            console.print(error_panel("Usage: /gh create <name> [--private]"))
            return
        name = args[1]
        private = "--private" in args
        info = gh.create_repo(name=name, private=private)
        console.print(success_panel(f"Created repo: {info.full_name}\n{info.url}"))

    # read file
    elif sub == "read":
        if len(args) < 3:
            console.print(error_panel("Usage: /gh read <repo> <path> [ref]"))
            return
        ref = args[3] if len(args) > 3 else ""
        fc = gh.read_file(args[1], args[2], ref)
        console.print(
            Panel(
                Markdown(f"```\n{fc.content}\n```"),
                title=f"[berome.code] {args[2]} [/berome.code]",
                border_style="cyan",
            )
        )

    # list directory
    elif sub == "ls":
        repo = args[1] if len(args) > 1 else ""
        path = args[2] if len(args) > 2 else ""
        if not repo:
            console.print(error_panel("Usage: /gh ls <repo> [path]"))
            return
        items = gh.list_directory(repo, path)
        _print_dir_listing(items, console)

    # write file
    elif sub == "write":
        if len(args) < 4:
            console.print(error_panel("Usage: /gh write <repo> <path> <commit-msg>"))
            return
        repo, path, msg = args[1], args[2], args[3]
        console.print(f"[berome.muted]Enter file content (end with a line containing only '.'):[/berome.muted]")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line == ".":
                break
            lines.append(line)
        content = "\n".join(lines)
        result = gh.create_or_update_file(repo, path, content, msg)
        console.print(success_panel(f"Committed: {result.sha[:7]}\n{result.url}"))

    # clone
    elif sub == "clone":
        if len(args) < 3:
            console.print(error_panel("Usage: /gh clone <repo> <local-dir>"))
            return
        from pathlib import Path
        cloned = gh.clone_repo(args[1], Path(args[2]))
        console.print(success_panel(f"Cloned to: {cloned}"))

    # push
    elif sub == "push":
        if len(args) < 3:
            console.print(error_panel("Usage: /gh push <local-dir> <commit-message>"))
            return
        from pathlib import Path
        msg = gh.commit_and_push(Path(args[1]), " ".join(args[2:]))
        console.print(success_panel(f"Pushed: {msg}"))

    # create branch
    elif sub == "branch":
        if len(args) < 3:
            console.print(error_panel("Usage: /gh branch <repo> <new-branch> [from-branch]"))
            return
        from_branch = args[3] if len(args) > 3 else ""
        branch = gh.create_branch(args[1], args[2], from_branch)
        console.print(success_panel(f"Branch created: {branch}"))

    # open PR
    elif sub == "pr":
        if len(args) < 4:
            console.print(error_panel("Usage: /gh pr <repo> <head-branch> <title>"))
            return
        pr = gh.create_pull_request(args[1], " ".join(args[3:]), "", args[2])
        console.print(success_panel(f"PR #{pr['number']} opened: {pr['url']}"))

    else:
        console.print(error_panel(f"Unknown gh sub-command: {sub}  — try /help"))


# ── Formatting helpers ────────────────────────────────────────────────────────


def _print_repo_detail(info, console: Console) -> None:
    from rich.table import Table
    from rich import box

    t = Table(box=box.SIMPLE, show_header=False)
    t.add_column("Key", style="bold cyan")
    t.add_column("Value")

    rows = [
        ("Name", info.full_name),
        ("URL", info.url),
        ("Private", "Yes" if info.private else "No"),
        ("Branch", info.default_branch),
        ("Stars", str(info.stars)),
        ("Forks", str(info.forks)),
        ("Description", info.description or "—"),
    ]
    for k, v in rows:
        t.add_row(k, v)
    console.print(Panel(t, title=f"[bold cyan]{info.name}[/bold cyan]", border_style="cyan"))


def _print_dir_listing(items: list[dict], console: Console) -> None:
    from rich.table import Table
    from rich import box

    t = Table(box=box.SIMPLE, show_header=True)
    t.add_column("Type", width=6)
    t.add_column("Name", style="bold")
    t.add_column("Size", justify="right", style="dim")

    for item in items:
        icon = "📁" if item["type"] == "dir" else "📄"
        t.add_row(icon, item["path"], str(item.get("size", "")) if item["type"] != "dir" else "")
    console.print(t)


if __name__ == "__main__":
    app()
