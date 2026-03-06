"""
Entry point for running the Berome Discord bot.

Usage::

    berome-discord            # reads DISCORD_BOT_TOKEN from .env

The bot can run independently of the CLI — they share the same installed
package but are separate processes.
"""
from __future__ import annotations

import logging
import sys


def main() -> None:
    """CLI entry point registered in pyproject.toml as ``berome-discord``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from berome.config import settings

    if not settings.discord_token:
        print(
            "Error: DISCORD_BOT_TOKEN is not set.\n"
            "Add it to your .env file:\n"
            "  DISCORD_BOT_TOKEN=your-bot-token-here\n\n"
            "Get a token at https://discord.com/developers/applications",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from berome.discord_bot import BeromeBot
    except ImportError:
        print(
            "Error: discord.py is not installed.\n"
            "Install it with:\n"
            "  pip install -e '.[discord]'",
            file=sys.stderr,
        )
        sys.exit(1)

    bot = BeromeBot()

    print(f"Starting Berome Discord bot...")
    print(f"Provider: {settings.provider.value} / {settings.active_model()}")
    print(f"Require @mention: {settings.discord_require_mention}")
    allowed = settings.discord_allowed_channel_ids()
    if allowed:
        print(f"Always-on channels: {', '.join(str(c) for c in allowed)}")
    print("Press Ctrl+C to stop.\n")

    bot.run(settings.discord_token, log_handler=None)


if __name__ == "__main__":
    main()
