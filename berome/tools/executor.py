"""
Tool executor — dispatches tool calls to concrete implementations.

All tool functions are async to allow clean composition with the agentic loop.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

from berome.tools.safety import is_safe_command


@dataclass
class ToolResult:
    tool_name: str
    tool_call_id: str
    output: str
    error: bool = False


# Type alias for the confirmation callback
ConfirmFn = Optional[Callable[[str], Awaitable[bool]]]


async def execute_tool(
    name: str,
    arguments: dict,
    tool_call_id: str = "",
    require_confirmation: ConfirmFn = None,
) -> ToolResult:
    """Dispatch a tool call by name and return a ToolResult."""
    dispatch = {
        "write_file": _write_file,
        "read_file": _read_file,
        "list_directory": _list_directory,
        "create_directory": _create_directory,
        "run_command": _run_command,
        "delete_file": _delete_file,
    }
    handler = dispatch.get(name)
    if handler is None:
        return ToolResult(
            tool_name=name,
            tool_call_id=tool_call_id,
            output=f"Unknown tool: {name!r}",
            error=True,
        )
    return await handler(arguments, tool_call_id, require_confirmation)


# ── Individual tool implementations ───────────────────────────────────────────


async def _write_file(
    args: dict, tool_call_id: str, _confirm: ConfirmFn
) -> ToolResult:
    path = args.get("path", "")
    content = args.get("content", "")
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        size = p.stat().st_size
        return ToolResult(
            tool_name="write_file",
            tool_call_id=tool_call_id,
            output=f"Written {size:,} bytes to {path}",
        )
    except Exception as exc:
        return ToolResult(
            tool_name="write_file",
            tool_call_id=tool_call_id,
            output=str(exc),
            error=True,
        )


async def _read_file(
    args: dict, tool_call_id: str, _confirm: ConfirmFn
) -> ToolResult:
    path = args.get("path", "")
    try:
        content = Path(path).read_text(encoding="utf-8")
        return ToolResult(
            tool_name="read_file",
            tool_call_id=tool_call_id,
            output=content,
        )
    except Exception as exc:
        return ToolResult(
            tool_name="read_file",
            tool_call_id=tool_call_id,
            output=str(exc),
            error=True,
        )


async def _list_directory(
    args: dict, tool_call_id: str, _confirm: ConfirmFn
) -> ToolResult:
    path = args.get("path", ".")
    try:
        p = Path(path)
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        lines = []
        for entry in entries:
            kind = "dir " if entry.is_dir() else "file"
            lines.append(f"[{kind}] {entry.name}")
        return ToolResult(
            tool_name="list_directory",
            tool_call_id=tool_call_id,
            output="\n".join(lines) if lines else "(empty directory)",
        )
    except Exception as exc:
        return ToolResult(
            tool_name="list_directory",
            tool_call_id=tool_call_id,
            output=str(exc),
            error=True,
        )


async def _create_directory(
    args: dict, tool_call_id: str, _confirm: ConfirmFn
) -> ToolResult:
    path = args.get("path", "")
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return ToolResult(
            tool_name="create_directory",
            tool_call_id=tool_call_id,
            output=f"Directory created: {path}",
        )
    except Exception as exc:
        return ToolResult(
            tool_name="create_directory",
            tool_call_id=tool_call_id,
            output=str(exc),
            error=True,
        )


async def _run_command(
    args: dict, tool_call_id: str, require_confirmation: ConfirmFn
) -> ToolResult:
    command = args.get("command", "")

    # Safety blocklist check
    safe, reason = is_safe_command(command)
    if not safe:
        return ToolResult(
            tool_name="run_command",
            tool_call_id=tool_call_id,
            output=f"Blocked: {reason}. Command not executed.",
            error=True,
        )

    # Ask for user confirmation
    if require_confirmation is not None:
        confirmed = await require_confirmation(command)
        if not confirmed:
            return ToolResult(
                tool_name="run_command",
                tool_call_id=tool_call_id,
                output="Command cancelled by user.",
                error=True,
            )

    # Execute
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = (result.stdout + result.stderr).strip() or "(no output)"
        return ToolResult(
            tool_name="run_command",
            tool_call_id=tool_call_id,
            output=output,
            error=result.returncode != 0,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            tool_name="run_command",
            tool_call_id=tool_call_id,
            output="Command timed out after 60 seconds.",
            error=True,
        )
    except Exception as exc:
        return ToolResult(
            tool_name="run_command",
            tool_call_id=tool_call_id,
            output=str(exc),
            error=True,
        )


async def _delete_file(
    args: dict, tool_call_id: str, _confirm: ConfirmFn
) -> ToolResult:
    path = args.get("path", "")
    try:
        p = Path(path)
        if not p.exists():
            return ToolResult(
                tool_name="delete_file",
                tool_call_id=tool_call_id,
                output=f"File not found: {path}",
                error=True,
            )
        if p.is_dir():
            return ToolResult(
                tool_name="delete_file",
                tool_call_id=tool_call_id,
                output=f"Path is a directory, not a file: {path}",
                error=True,
            )
        p.unlink()
        return ToolResult(
            tool_name="delete_file",
            tool_call_id=tool_call_id,
            output=f"Deleted: {path}",
        )
    except Exception as exc:
        return ToolResult(
            tool_name="delete_file",
            tool_call_id=tool_call_id,
            output=str(exc),
            error=True,
        )
