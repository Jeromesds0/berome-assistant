"""Safety checks for the run_command tool."""

from __future__ import annotations

import re

# Each entry is (regex_pattern, human_reason)
_BLOCKLIST: list[tuple[str, str]] = [
    (r"rm\s+-[a-zA-Z]*r[a-zA-Z]*f\s*/(?:\s|$)", "rm -rf /"),
    (r":\(\)\s*\{.*:\|:.*\}", "fork bomb"),
    (r"\bmkfs\b", "mkfs (destroys filesystem)"),
    (r"\bdd\b.*\bif=/dev/(?:zero|random|urandom)", "dd with device input"),
    (r">\s*/dev/sd[a-z]", "writing directly to block device"),
    (r"\bchmod\s+[0-7]*7[0-7]*\s+/(?:\s|$)", "chmod 777 on root"),
    (r"wget\s+\S+\s*\|\s*(?:ba)?sh", "wget pipe to shell"),
    (r"curl\s+\S+\s*\|\s*(?:ba)?sh", "curl pipe to shell"),
    (r"\bsudo\s+rm\s+-[a-zA-Z]*r[a-zA-Z]*f", "sudo rm -rf"),
    (r">\s*/dev/zero", "write to /dev/zero"),
    (r"\bshred\b.*\s+-[a-zA-Z]*u", "shred + delete"),
    (r"\bpoweroff\b|\breboot\b|\bshutdown\b", "system power control"),
    (r"\bmv\b.*\s+/dev/null", "move files to /dev/null"),
]


def is_safe_command(command: str) -> tuple[bool, str]:
    """Return ``(True, "")`` if safe, ``(False, reason)`` if blocked."""
    for pattern, reason in _BLOCKLIST:
        if re.search(pattern, command, re.IGNORECASE):
            return False, reason
    return True, ""
