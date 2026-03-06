"""
Persistent per-guild storage for Berome Discord bot.

Stores:
- memories: list of facts taught via /teach
- active_channels: set of channel IDs that respond without @mention

Data lives in ~/.berome/guilds/{guild_id}.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_DIR = Path.home() / ".berome" / "guilds"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

_ACTIVE_CHANNELS_PATH = Path.home() / ".berome" / "active_channels.json"


# ── Guild memories ─────────────────────────────────────────────────────────────


def _guild_path(guild_id: int) -> Path:
    return _DATA_DIR / f"{guild_id}.json"


def _load_guild(guild_id: int) -> dict:
    p = _guild_path(guild_id)
    if not p.exists():
        return {"memories": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"memories": []}


def _save_guild(guild_id: int, data: dict) -> None:
    _guild_path(guild_id).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_memories(guild_id: int) -> list[str]:
    return _load_guild(guild_id).get("memories", [])


def add_memory(guild_id: int, fact: str) -> list[str]:
    data = _load_guild(guild_id)
    data.setdefault("memories", []).append(fact)
    _save_guild(guild_id, data)
    return data["memories"]


def remove_memory(guild_id: int, index: int) -> tuple[str, list[str]]:
    data = _load_guild(guild_id)
    memories: list[str] = data.setdefault("memories", [])
    removed = memories.pop(index)
    _save_guild(guild_id, data)
    return removed, memories


# ── Active channels ────────────────────────────────────────────────────────────


def load_active_channels() -> set[int]:
    if not _ACTIVE_CHANNELS_PATH.exists():
        return set()
    try:
        return set(json.loads(_ACTIVE_CHANNELS_PATH.read_text(encoding="utf-8")))
    except Exception:
        return set()


def save_active_channels(channel_ids: set[int]) -> None:
    _ACTIVE_CHANNELS_PATH.write_text(
        json.dumps(list(channel_ids)), encoding="utf-8"
    )
