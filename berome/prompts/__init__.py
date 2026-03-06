"""Loads system prompt files from this directory."""
from pathlib import Path

_DIR = Path(__file__).parent


def load(name: str) -> str:
    """Read and return a prompt file, stripping surrounding whitespace."""
    return (_DIR / name).read_text(encoding="utf-8").strip()
