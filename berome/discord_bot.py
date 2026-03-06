"""
Discord bot frontend for Berome.

Architecture:
- One BeromeSession per Discord channel (keyed by channel_id).
- Sessions are in-memory; cleared on /clear or bot restart.
- Sessions idle for >1 hour are automatically evicted.
- agentic_stream() is used when the provider supports chat_with_tools.
- Responses are collected fully, then sent (edit-after-complete strategy).
- Shell command confirmations use discord.ui.View with Confirm/Cancel buttons.

Usage:
    from berome.discord_bot import BeromeBot
    bot = BeromeBot()
    bot.run(token)
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import discord
from discord import app_commands

from berome.config import settings
from berome.prompts import load as _load_prompt
from berome.session import BeromeSession

logger = logging.getLogger(__name__)

DISCORD_MAX_LENGTH = 2000
SESSION_TTL = 3600  # seconds — evict sessions idle longer than this
HISTORY_SEED_LIMIT = 50  # number of recent channel messages to load into context

DISCORD_SYSTEM_PROMPT = _load_prompt("discord_system.md")

TOOL_EMOJI: dict[str, str] = {
    "write_file": "📝",
    "read_file": "📖",
    "list_directory": "📁",
    "create_directory": "📂",
    "run_command": "⚙️",
    "delete_file": "🗑️",
    "web_search": "🔍",
}


# ── Confirmation UI ────────────────────────────────────────────────────────────


class ConfirmView(discord.ui.View):
    """
    A two-button Confirm/Cancel view for shell command authorization.

    Usage::

        view = ConfirmView(command="ls -la", timeout=60.0)
        await channel.send(f"Run `{command}`?", view=view)
        confirmed = await view.wait_for_decision()
    """

    def __init__(self, command: str, timeout: float = 60.0) -> None:
        super().__init__(timeout=timeout)
        self.command = command
        self._decision: asyncio.Future[bool] = asyncio.get_event_loop().create_future()

    @discord.ui.button(label="Confirm", style=discord.ButtonStyle.danger, emoji="✅")
    async def confirm(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        await interaction.response.edit_message(
            content=f"Running: `{self.command[:1800]}`", view=None
        )
        self.stop()
        if not self._decision.done():
            self._decision.set_result(True)

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary, emoji="❌")
    async def cancel(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        await interaction.response.edit_message(
            content=f"Cancelled: `{self.command[:1800]}`", view=None
        )
        self.stop()
        if not self._decision.done():
            self._decision.set_result(False)

    async def on_timeout(self) -> None:
        if not self._decision.done():
            self._decision.set_result(False)

    async def wait_for_decision(self) -> bool:
        """Block until a button is clicked or the timeout expires."""
        return await self._decision


# ── Main bot class ─────────────────────────────────────────────────────────────


class BeromeBot(discord.Client):
    """
    Discord bot wrapping per-channel BeromeSession instances.

    Uses discord.Client (not commands.Bot) to avoid the internal on_message
    listener that commands.Bot registers for process_commands, which would
    cause every message to be handled twice.

    Session lifecycle:
    - Sessions are created lazily on first message in a channel.
    - Key: channel.id (int). Works for DMChannel, TextChannel, and Thread.
    - Threads get their own session (thread.id differs from parent channel.id).
    - Sessions idle for SESSION_TTL seconds are evicted to prevent memory growth.
    """

    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True  # Required to read message text
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self._sessions: dict[int, BeromeSession] = {}
        self._session_last_used: dict[int, float] = {}

    # ── Session management ─────────────────────────────────────────────────

    async def _get_session(self, channel: discord.abc.Messageable) -> Optional[BeromeSession]:
        """Return (or create) a BeromeSession for the given channel.

        On first creation, seeds the session with recent channel history so the
        bot has context about how people speak and what was already discussed.
        """
        channel_id = channel.id  # type: ignore[attr-defined]
        self._evict_stale_sessions()
        self._session_last_used[channel_id] = time.monotonic()
        if channel_id not in self._sessions:
            try:
                logger.info("Creating new BeromeSession for channel %d", channel_id)
                session = BeromeSession(system_prompt=DISCORD_SYSTEM_PROMPT)
                await self._seed_history(session, channel)
                self._sessions[channel_id] = session
            except RuntimeError as exc:
                logger.error(
                    "Failed to create session for channel %d: %s", channel_id, exc
                )
                return None
        return self._sessions[channel_id]

    async def _seed_history(
        self, session: BeromeSession, channel: discord.abc.Messageable
    ) -> None:
        """Load recent messages from the channel into the session history.

        This gives the bot awareness of ongoing conversation style and context.
        Bot's own messages are added as 'assistant'; others as 'user'.
        Tool-status messages (Thinking..., emoji previews) are skipped.
        """
        _tool_prefixes = tuple(TOOL_EMOJI.values()) + ("*Thinking...*",)
        messages: list[tuple[str, str]] = []
        try:
            async for msg in channel.history(  # type: ignore[attr-defined]
                limit=HISTORY_SEED_LIMIT, oldest_first=True
            ):
                if msg.author.bot:
                    if self.user and msg.author.id == self.user.id:
                        # Our own message — skip tool status updates
                        if not any(msg.content.startswith(p) for p in _tool_prefixes):
                            messages.append(("assistant", msg.content))
                    # Ignore other bots
                else:
                    content = msg.content
                    if self.user:
                        content = content.replace(f"<@{self.user.id}>", "")
                        content = content.replace(f"<@!{self.user.id}>", "")
                    content = content.strip()
                    if content:
                        # Prefix with display name so the LLM sees per-user style
                        display = msg.author.display_name
                        messages.append(("user", f"[{display}]: {content}"))
        except (discord.Forbidden, discord.HTTPException) as exc:
            logger.warning("Could not load channel history for seeding: %s", exc)
            return

        for role, content in messages:
            session.add_history_message(role, content)

        if messages:
            logger.info(
                "Seeded session for channel %s with %d messages",
                getattr(channel, "id", "?"),
                len(messages),
            )

    def _evict_stale_sessions(self) -> None:
        """Remove sessions that have been idle for longer than SESSION_TTL."""
        now = time.monotonic()
        stale = [
            cid
            for cid, t in self._session_last_used.items()
            if now - t > SESSION_TTL
        ]
        for cid in stale:
            self._sessions.pop(cid, None)
            self._session_last_used.pop(cid, None)
            logger.info("Evicted stale session for channel %d", cid)

    # ── Bot lifecycle ──────────────────────────────────────────────────────

    async def setup_hook(self) -> None:
        """Register slash commands and sync them to Discord."""
        self.tree.add_command(_ClearCommand(self))
        self.tree.add_command(_StatusCommand(self))
        self.tree.add_command(_ProviderCommand(self))
        self.tree.add_command(_HelpCommand(self))
        await self.tree.sync()
        logger.info("Slash commands synced.")

    async def on_ready(self) -> None:
        logger.info("Berome Discord bot ready. Logged in as %s (ID: %s)", self.user, self.user.id if self.user else "?")
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name="your questions",
            )
        )

    # ── Message routing ────────────────────────────────────────────────────

    async def on_message(self, message: discord.Message) -> None:
        # Never respond to ourselves
        if message.author == self.user:
            return

        if not self._should_respond(message):
            return

        user_text = self._extract_text(message)
        if not user_text:
            return

        await self._handle_message(message, user_text)

    def _should_respond(self, message: discord.Message) -> bool:
        """
        Return True if the bot should respond to this message.

        Priority order:
        1. DMs → always respond.
        2. Channel in DISCORD_ALLOWED_CHANNELS → always respond.
        3. DISCORD_REQUIRE_MENTION=false → always respond.
        4. Bot is mentioned → respond.
        5. Otherwise → ignore.
        """
        if isinstance(message.channel, discord.DMChannel):
            return True

        channel_id = message.channel.id
        if channel_id in settings.discord_allowed_channel_ids():
            return True

        if not settings.discord_require_mention:
            return True

        if self.user and self.user.mentioned_in(message):
            return True

        return False

    def _extract_text(self, message: discord.Message) -> str:
        """Strip bot @mention prefix and return clean user text."""
        text = message.content
        if self.user:
            text = text.replace(f"<@{self.user.id}>", "")
            text = text.replace(f"<@!{self.user.id}>", "")
        return text.strip()

    # ── Core response handler ──────────────────────────────────────────────

    async def _handle_message(self, message: discord.Message, user_text: str) -> None:
        """
        Process one user message and send the bot's reply.

        Uses Discord's native typing indicator while generating, then sends
        the response as a new message. No placeholder message is shown.
        """
        channel = message.channel
        session = await self._get_session(channel)
        if session is None:
            await channel.send(
                "Configuration error: the LLM provider is not available. "
                "Check that `ANTHROPIC_API_KEY` (or Ollama) is set correctly.",
                reference=message,
            )
            return

        async def on_tool_call(name: str, args: dict) -> None:
            # Typing indicator is already showing; no extra message needed
            pass

        async def on_tool_result(result) -> None:
            if result.error:
                try:
                    await channel.send(
                        f"Tool error (`{result.tool_name}`): {result.output[:400]}",
                        reference=message,
                    )
                except discord.HTTPException:
                    pass

        async def require_confirmation(command: str) -> bool:
            if settings.discord_auto_approve_tools:
                return True
            view = ConfirmView(command=command, timeout=60.0)
            try:
                await channel.send(
                    f"**Shell command requested:**\n```\n{command[:1500]}\n```\nConfirm execution?",
                    view=view,
                )
            except discord.HTTPException:
                return False
            return await view.wait_for_decision()

        response_parts: list[str] = []
        try:
            # channel.typing() shows Discord's native "Bot is typing..." indicator
            # and auto-refreshes every 5 seconds for as long as the block runs.
            async with channel.typing():
                if hasattr(session.provider, "chat_with_tools"):
                    async for chunk in session.agentic_stream(
                        user_text, on_tool_call, on_tool_result, require_confirmation
                    ):
                        response_parts.append(chunk)
                else:
                    async for chunk in session.chat_stream(user_text):
                        response_parts.append(chunk)
        except Exception as exc:
            logger.exception("Error during stream for channel %d", channel.id)
            await channel.send(f"An error occurred: {str(exc)[:300]}", reference=message)
            return

        full_response = "".join(response_parts).strip()
        if not full_response:
            return

        await _send_response(full_response, channel)


# ── Slash commands ─────────────────────────────────────────────────────────────


class _ClearCommand(app_commands.Command):
    """Clear conversation history for this channel."""

    def __init__(self, bot: BeromeBot) -> None:
        self._bot = bot
        super().__init__(
            name="clear",
            description="Clear Berome's conversation history for this channel",
            callback=self._callback,
        )

    async def _callback(self, interaction: discord.Interaction) -> None:
        channel_id = interaction.channel_id
        if channel_id and channel_id in self._bot._sessions:
            self._bot._sessions[channel_id].clear_history()
            await interaction.response.send_message(
                "Conversation history cleared for this channel.", ephemeral=True
            )
        else:
            await interaction.response.send_message(
                "No conversation history to clear.", ephemeral=True
            )


class _StatusCommand(app_commands.Command):
    """Show active sessions and token usage."""

    def __init__(self, bot: BeromeBot) -> None:
        self._bot = bot
        super().__init__(
            name="status",
            description="Show Berome's active sessions and token usage",
            callback=self._callback,
        )

    async def _callback(self, interaction: discord.Interaction) -> None:
        channel_id = interaction.channel_id
        session_count = len(self._bot._sessions)
        lines = [f"**Active sessions:** {session_count}"]

        if channel_id and channel_id in self._bot._sessions:
            sess = self._bot._sessions[channel_id]
            stats = sess.token_stats()
            lines += [
                "**This channel:**",
                f"  Provider: `{sess.provider.provider_name}` / `{sess.provider.model_name}`",
                f"  Session tokens: {stats['session_in']} in / {stats['session_out']} out",
                f"  Last response: {stats['last_in']} in / {stats['last_out']} out",
            ]
        else:
            lines.append("No active session in this channel yet.")

        await interaction.response.send_message("\n".join(lines), ephemeral=True)


class _ProviderCommand(app_commands.Command):
    """Show or switch the LLM provider for this channel."""

    def __init__(self, bot: BeromeBot) -> None:
        self._bot = bot
        super().__init__(
            name="provider",
            description="Show or switch the LLM provider for this channel",
            callback=self._callback,
        )

    async def _callback(
        self,
        interaction: discord.Interaction,
        name: Optional[str] = None,
    ) -> None:
        channel_id = interaction.channel_id
        # Use existing session if present; create lazily only when switching
        session = self._bot._sessions.get(channel_id) if channel_id else None
        if session is None and name is not None:
            session = await self._bot._get_session(interaction.channel)
        if session is None:
            await interaction.response.send_message(
                "Provider unavailable — check your LLM configuration.", ephemeral=True
            )
            return

        if name is None:
            await interaction.response.send_message(
                f"Provider: `{session.provider.provider_name}` / `{session.provider.model_name}`",
                ephemeral=True,
            )
            return

        try:
            session.switch_provider(name)
            await interaction.response.send_message(
                f"Switched to `{name}` for this channel.", ephemeral=True
            )
        except Exception as exc:
            await interaction.response.send_message(
                f"Failed to switch provider: {exc}", ephemeral=True
            )


class _HelpCommand(app_commands.Command):
    """Show usage and available commands."""

    def __init__(self, bot: BeromeBot) -> None:
        self._bot = bot
        super().__init__(
            name="berome-help",
            description="Show Berome bot help",
            callback=self._callback,
        )

    async def _callback(self, interaction: discord.Interaction) -> None:
        help_text = (
            "**Berome Discord Bot**\n\n"
            "**Usage:**\n"
            "- Mention me (`@Berome <message>`) to chat\n"
            "- In DMs, just type freely\n"
            "- In channels listed in `DISCORD_ALLOWED_CHANNELS`, type without mention\n\n"
            "**Slash Commands:**\n"
            "`/clear` — Clear conversation history for this channel\n"
            "`/status` — Show session info and token usage\n"
            "`/provider [name]` — Show or switch LLM provider (`anthropic` / `ollama`)\n"
            "`/berome-help` — Show this help\n\n"
            "**Tools available:**\n"
            "📖 read_file  📝 write_file  📁 list_directory  📂 create_directory\n"
            "⚙️ run_command *(requires confirmation)*  🗑️ delete_file"
        )
        await interaction.response.send_message(help_text, ephemeral=True)


# ── Utility functions ──────────────────────────────────────────────────────────


def _format_args_preview(args: dict, max_len: int = 80) -> str:
    """Format tool arguments as a short preview string."""
    parts = []
    for k, v in args.items():
        v_str = repr(v)
        if len(v_str) > 40:
            v_str = v_str[:37] + "..."
        parts.append(f"{k}={v_str}")
    result = ", ".join(parts)
    return result[:max_len] + "..." if len(result) > max_len else result


async def _send_response(content: str, channel: discord.abc.Messageable) -> None:
    """
    Send the full response, splitting across multiple messages if over 2000 chars.
    """
    for chunk in _split_message(content):
        try:
            await channel.send(chunk)
        except discord.HTTPException as exc:
            logger.warning("Failed to send response chunk: %s", exc)


def _split_message(text: str, limit: int = DISCORD_MAX_LENGTH) -> list[str]:
    """
    Split text into chunks of at most `limit` characters.

    Splits preferentially at newline boundaries to preserve markdown.
    Falls back to a hard split if a single line exceeds the limit.
    Tracks open triple-backtick fences and closes/reopens them across chunks
    so code blocks are not broken mid-render.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break

        # Try to split at a newline within the limit
        split_pos = text.rfind("\n", 0, limit)
        if split_pos == -1:
            split_pos = limit

        chunk = text[:split_pos]
        text = text[split_pos:].lstrip("\n")

        # Check if we cut inside a code fence (odd number of ``` up to split_pos)
        fence_count = chunk.count("```")
        if fence_count % 2 == 1:
            # We're inside an open fence — close it in this chunk and reopen next
            chunk = chunk + "\n```"
            text = "```\n" + text

        chunks.append(chunk)

    return chunks
