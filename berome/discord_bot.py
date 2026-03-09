"""
Discord bot frontend for Berome.

Architecture:
- One BeromeSession per Discord channel (keyed by channel_id).
- Sessions are in-memory; cleared on /clear or bot restart.
- Sessions idle for >1 hour are automatically evicted.
- agentic_stream() is used when the provider supports chat_with_tools.
- Images in messages are downloaded, base64-encoded, and passed to the LLM.
- Shell command confirmations use discord.ui.View with Confirm/Cancel buttons.
- Guild memories (/teach, /forget) are persisted across restarts.
- Active channels (/activate, /deactivate) respond without @mention.

Usage:
    from berome.discord_bot import BeromeBot
    bot = BeromeBot()
    bot.run(token)
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from pathlib import Path
from typing import Optional

import discord
import httpx
from discord import app_commands

from berome.config import settings
from berome.guild_data import (
    add_memory,
    load_active_channels,
    load_memories,
    remove_memory,
    save_active_channels,
)
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
    "image_search": "🖼️",
}

# MIME types accepted for image analysis
_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}


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
        intents.members = True  # Required to list guild members for @mentions
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self._sessions: dict[int, BeromeSession] = {}
        self._session_last_used: dict[int, float] = {}
        self._active_channels: set[int] = load_active_channels()
        self._channel_locks: dict[int, asyncio.Lock] = {}

    # ── Session management ─────────────────────────────────────────────────

    async def _get_session(self, channel: discord.abc.Messageable) -> Optional[BeromeSession]:
        """Return (or create) a BeromeSession for the given channel.

        On first creation, seeds the session with recent channel history and
        any guild memories that have been taught via /teach.
        """
        channel_id = channel.id  # type: ignore[attr-defined]
        self._evict_stale_sessions()
        self._session_last_used[channel_id] = time.monotonic()
        if channel_id not in self._sessions:
            try:
                logger.info("Creating new BeromeSession for channel %d", channel_id)
                system_prompt = self._build_system_prompt(channel)
                session = BeromeSession(system_prompt=system_prompt)
                await self._seed_members(session, channel)
                await self._seed_history(session, channel)
                self._sessions[channel_id] = session
            except RuntimeError as exc:
                logger.error(
                    "Failed to create session for channel %d: %s", channel_id, exc
                )
                return None
        return self._sessions[channel_id]

    def _build_system_prompt(self, channel: discord.abc.Messageable) -> str:
        """Build a system prompt that includes any guild memories."""
        guild_id = getattr(getattr(channel, "guild", None), "id", None)
        if guild_id is None:
            return DISCORD_SYSTEM_PROMPT
        memories = load_memories(guild_id)
        if not memories:
            return DISCORD_SYSTEM_PROMPT
        facts = "\n".join(f"- {m}" for m in memories)
        return (
            DISCORD_SYSTEM_PROMPT
            + f"\n\n**Facts you've been taught about this server:**\n{facts}"
        )

    async def _seed_members(
        self, session: BeromeSession, channel: discord.abc.Messageable
    ) -> None:
        """Inject server member list so the bot knows how to @mention people."""
        guild = getattr(channel, "guild", None)
        if guild is None:
            return  # DM — no members to inject
        try:
            members = [m for m in guild.members if not m.bot]
            if not members:
                return
            lines = [
                f"- {m.display_name} (username: {m.name}, mention: <@{m.id}>)"
                for m in members
            ]
            member_list = "\n".join(lines)
            session.add_history_message(
                "user",
                f"[System]: Server members you can @mention (use the exact mention format shown):\n{member_list}",
            )
            session.add_history_message(
                "assistant",
                "Got it. I know the server members and their mention formats.",
            )
        except Exception as exc:
            logger.warning("Could not seed member list: %s", exc)

    async def _seed_history(
        self, session: BeromeSession, channel: discord.abc.Messageable
    ) -> None:
        """Load recent messages from the channel into the session history.

        This gives the bot awareness of ongoing conversation style and context.
        Bot's own messages are added as 'assistant'; others as 'user'.
        Tool-status messages are skipped.
        """
        _tool_prefixes = tuple(TOOL_EMOJI.values())
        messages: list[tuple[str, str]] = []
        try:
            async for msg in channel.history(  # type: ignore[attr-defined]
                limit=HISTORY_SEED_LIMIT, oldest_first=True
            ):
                if msg.author.bot:
                    if self.user and msg.author.id == self.user.id:
                        if not any(msg.content.startswith(p) for p in _tool_prefixes):
                            messages.append(("assistant", msg.content))
                else:
                    content = msg.content
                    if self.user:
                        content = content.replace(f"<@{self.user.id}>", "")
                        content = content.replace(f"<@!{self.user.id}>", "")
                    content = content.strip()
                    if content:
                        display = msg.author.display_name
                        messages.append(("user", f"[{display}]: {content}"))
        except (discord.Forbidden, discord.HTTPException) as exc:
            logger.warning("Could not load channel history for seeding: %s", exc)
            return

        for role, content in messages:
            session.add_history_message(role, content)

        # Always inject an English reminder after seeding so that a
        # non-English chat history doesn't cause the bot to reply in the
        # wrong language.
        session.add_history_message(
            "user",
            "[System]: Reminder — always reply in English, no matter what language the messages above are in.",
        )
        session.add_history_message("assistant", "Understood, I'll always reply in English.")

        if messages:
            logger.info(
                "Seeded session for channel %s with %d messages",
                getattr(channel, "id", "?"),
                len(messages),
            )

    async def _seed_other_channels(
        self, session: BeromeSession, current_channel: discord.abc.Messageable
    ) -> None:
        """Seed recent messages from every other readable text channel in the guild.

        Gives the bot server-wide context so it can answer questions like
        "what was said in #general?" or "do you know about the conversation in #dev?".
        Limited to the last 15 messages per channel to keep context manageable.
        """
        guild = getattr(current_channel, "guild", None)
        if guild is None:
            return  # DM — no other channels

        current_id = getattr(current_channel, "id", None)
        _tool_prefixes = tuple(TOOL_EMOJI.values())
        cross: list[tuple[str, str]] = []

        for ch in guild.text_channels:
            if ch.id == current_id:
                continue  # already seeded above
            try:
                ch_msgs: list[tuple[str, str]] = []
                async for msg in ch.history(limit=15, oldest_first=False):
                    if msg.author.bot:
                        if self.user and msg.author.id == self.user.id:
                            if not any(msg.content.startswith(p) for p in _tool_prefixes):
                                ch_msgs.append(("assistant", f"[#{ch.name}] {msg.content}"))
                    else:
                        content = msg.content
                        if self.user:
                            content = content.replace(f"<@{self.user.id}>", "")
                            content = content.replace(f"<@!{self.user.id}>", "")
                        content = content.strip()
                        if content:
                            display = msg.author.display_name
                            ch_msgs.append(("user", f"[#{ch.name}] [{display}]: {content}"))
                # Reverse so messages are chronological
                cross.extend(reversed(ch_msgs))
            except (discord.Forbidden, discord.HTTPException):
                pass  # Skip channels the bot can't read

        if not cross:
            return

        session.add_history_message(
            "user",
            "[System]: Below are recent messages from other channels in this server. "
            "You can reference them when asked about conversations in those channels.",
        )
        session.add_history_message(
            "assistant",
            "Got it — I can see the recent history from all the other channels.",
        )
        for role, content in cross:
            session.add_history_message(role, content)

        logger.info(
            "Seeded cross-channel history: %d messages from %d channel(s)",
            len(cross),
            len({m[1].split("]")[0] for m in cross}),
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
        self.tree.add_command(_ActivateCommand(self))
        self.tree.add_command(_DeactivateCommand(self))
        self.tree.add_command(_TeachCommand(self))
        self.tree.add_command(_ForgetCommand(self))
        self.tree.add_command(_MemoriesCommand(self))
        self.tree.add_command(_RoleGiveCommand(self))
        self.tree.add_command(_RoleTakeCommand(self))
        self.tree.add_command(_RoleCreateCommand(self))
        self.tree.add_command(_RoleDeleteCommand(self))
        self.tree.add_command(_RoleListCommand(self))
        self.tree.add_command(_DocumentCommand(self))
        self.tree.add_command(_SearchCommand(self))
        await self.tree.sync()
        logger.info("Slash commands synced.")

    async def on_ready(self) -> None:
        logger.info(
            "Berome Discord bot ready. Logged in as %s (ID: %s)",
            self.user,
            self.user.id if self.user else "?",
        )
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

        should_respond = self._should_respond(message)

        # Passively record human messages into existing sessions so the bot
        # stays aware of the full conversation, even when not @mentioned.
        if not should_respond and not message.author.bot:
            channel_id = message.channel.id
            if channel_id in self._sessions:
                text = self._extract_text(message)
                if text:
                    display = message.author.display_name
                    self._sessions[channel_id].add_history_message(
                        "user", f"[{display}]: {text}"
                    )
            return

        if not should_respond:
            return

        user_text = self._extract_text(message)
        images = await self._extract_images(message)

        # If the user replied to a message, extract images from that message too.
        # This lets users reply to a chart/graph and ask the bot to analyse it.
        # Always use fetch_message — message.reference.resolved may have incomplete
        # attachment data from the gateway event (missing content_type / CDN URLs).
        if message.reference is not None and message.reference.message_id is not None:
            try:
                ref_msg = await message.channel.fetch_message(  # type: ignore[attr-defined]
                    message.reference.message_id
                )
                ref_images = await self._extract_images(ref_msg)
                # Prepend so the referenced image appears before the current message's images
                images = ref_images + images
                if ref_images:
                    logger.info(
                        "Extracted %d image(s) from referenced message %d",
                        len(ref_images),
                        message.reference.message_id,
                    )
            except Exception as exc:
                logger.warning("Could not fetch referenced message for images: %s", exc)

        # Ignore messages with no text AND no images
        if not user_text and not images:
            return

        await self._handle_message(message, user_text, images)

    def _should_respond(self, message: discord.Message) -> bool:
        """
        Return True if the bot should respond to this message.

        Priority order:
        1. DMs → always respond.
        2. Channel in _active_channels → always respond.
        3. Channel in DISCORD_ALLOWED_CHANNELS → always respond.
        4. DISCORD_REQUIRE_MENTION=false → always respond.
        5. Bot is mentioned → respond.
        6. Otherwise → ignore.
        """
        if isinstance(message.channel, discord.DMChannel):
            return True

        channel_id = message.channel.id

        if channel_id in self._active_channels:
            return True

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

    async def _extract_images(self, message: discord.Message) -> list[dict]:
        """Download image attachments and embeds, return Anthropic content blocks."""
        urls: list[str] = []

        # Direct file attachments
        _EXT_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                     ".gif": "image/gif", ".webp": "image/webp"}
        for attachment in message.attachments:
            mime = (attachment.content_type or "").split(";")[0].strip()
            if mime not in _IMAGE_MIME_TYPES:
                # Fallback: infer from filename extension
                mime = _EXT_MIME.get(Path(attachment.filename).suffix.lower(), "")
            if mime in _IMAGE_MIME_TYPES:
                urls.append(attachment.url)

        # Embedded images (e.g. images posted as embeds by bots or Discord auto-embeds)
        for embed in message.embeds:
            if embed.image and embed.image.url:
                urls.append(embed.image.url)
            elif embed.thumbnail and embed.thumbnail.url:
                urls.append(embed.thumbnail.url)
            elif embed.type == "image" and embed.url:
                urls.append(embed.url)

        blocks: list[dict] = []
        async with httpx.AsyncClient(timeout=15) as client:
            for url in urls:
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    content_type = resp.headers.get("content-type", "image/png")
                    media_type = content_type.split(";")[0].strip()
                    if media_type not in _IMAGE_MIME_TYPES:
                        media_type = "image/png"
                    data = base64.standard_b64encode(resp.content).decode()
                    blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    })
                except Exception as exc:
                    logger.warning("Failed to download image from %s: %s", url, exc)
        return blocks

    # ── Core response handler ──────────────────────────────────────────────

    async def _handle_message(
        self,
        message: discord.Message,
        user_text: str,
        images: list[dict],
    ) -> None:
        """
        Process one user message (optionally with images) and send the reply.

        When images are present, builds a multimodal content block list and
        injects it directly into the session history, then calls
        continue_agentic_stream() so no duplicate user message is added.

        A per-channel asyncio.Lock serializes concurrent calls so rapid
        back-to-back messages never corrupt the same session's history.
        """
        channel = message.channel
        channel_id = channel.id  # type: ignore[attr-defined]
        lock = self._channel_locks.setdefault(channel_id, asyncio.Lock())
        async with lock:
            await self._handle_message_locked(message, user_text, images)

    async def _handle_message_locked(
        self,
        message: discord.Message,
        user_text: str,
        images: list[dict],
    ) -> None:
        """Inner handler — called under the per-channel lock."""
        channel = message.channel
        session = await self._get_session(channel)
        if session is None:
            await channel.send(
                "Configuration error: the LLM provider is not available. "
                "Check that `ANTHROPIC_API_KEY` (or Ollama) is set correctly.",
                reference=message,
            )
            return

        # Track files written by write_file tool so we can attach them
        _pending_write_path: list[str] = []  # filled by on_tool_call
        written_docs: list[Path] = []

        _TOOL_STATUS: dict[str, str] = {
            "web_search": "🔍 Searching: **{query}**",
            "image_search": "🖼️ Image search: **{query}**",
            "read_file": "📖 Reading: `{path}`",
            "write_file": "📝 Writing: `{path}`",
            "run_command": "⚙️ Running: `{command}`",
            "list_directory": "📁 Listing: `{path}`",
            "delete_file": "🗑️ Deleting: `{path}`",
        }

        async def on_tool_call(name: str, args: dict) -> None:
            if name == "write_file":
                raw = args.get("path", "") or "output.md"
                p = Path(raw)
                # Mirror the executor's sanitisation: dir → dir/output.md
                if p.is_dir():
                    p = p / "output.md"
                _pending_write_path.append(str(p))
            # Post a visible status line so users can see the bot working
            template = _TOOL_STATUS.get(name)
            if template:
                try:
                    status = template.format_map({k: str(v)[:80] for k, v in args.items()})
                    await channel.send(status)
                except (discord.HTTPException, KeyError):
                    pass

        async def on_tool_result(result) -> None:
            if result.tool_name == "write_file" and not result.error and _pending_write_path:
                p = Path(_pending_write_path.pop(0))
                if p.exists() and p.suffix.lower() in {".md", ".html", ".txt", ".pdf"}:
                    written_docs.append(p)
            if result.error:
                try:
                    await channel.send(
                        f"⚠️ Tool error (`{result.tool_name}`): {result.output[:400]}",
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
            async with channel.typing():
                display = message.author.display_name
                if images:
                    # Build multimodal content block (images + optional text)
                    content_blocks: list[dict] = list(images)
                    prompt = f"[{display}]: {user_text}" if user_text else f"[{display}]: What's in this image?"
                    content_blocks.append({"type": "text", "text": prompt})
                    session.add_history_message("user", content_blocks)
                    async for chunk in session.continue_agentic_stream(
                        on_tool_call, on_tool_result, require_confirmation
                    ):
                        response_parts.append(chunk)
                elif hasattr(session.provider, "chat_with_tools"):
                    async for chunk in session.agentic_stream(
                        f"[{display}]: {user_text}", on_tool_call, on_tool_result, require_confirmation
                    ):
                        response_parts.append(chunk)
                else:
                    async for chunk in session.chat_stream(f"[{display}]: {user_text}"):
                        response_parts.append(chunk)
        except Exception as exc:
            logger.exception("Error during stream for channel %d", channel.id)
            await channel.send(f"An error occurred: {str(exc)[:300]}", reference=message)
            return

        full_response = "".join(response_parts).strip()
        if not full_response and not written_docs:
            return

        # If the bot produced a long text response (likely a document) but didn't
        # write a file, save it automatically and attach it.
        _DOC_KEYWORDS = {"essay", "document", "report", "research", "write", "article"}
        is_doc_request = bool(_DOC_KEYWORDS & set(user_text.lower().split()))
        if full_response and not written_docs and is_doc_request and len(full_response) > 600:
            slug = "_".join(user_text.lower().split()[:5])[:40]
            auto_path = Path(f"{slug}.md")
            auto_path.write_text(full_response, encoding="utf-8")
            written_docs.append(auto_path)
            full_response = ""  # don't also dump it as chat

        if full_response:
            await _send_response(full_response, channel)

        # Attach any documents the bot wrote during the tool loop
        if written_docs:
            await _send_documents(written_docs, channel)


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
        active_count = len(self._bot._active_channels)
        lines = [
            f"**Active sessions:** {session_count}",
            f"**Active channels (no-mention mode):** {active_count}",
        ]

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


class _ActivateCommand(app_commands.Command):
    """Make the bot respond to all messages in this channel (no @mention needed)."""

    def __init__(self, bot: BeromeBot) -> None:
        self._bot = bot
        super().__init__(
            name="activate",
            description="Berome will respond to all messages in this channel (no @mention needed)",
            callback=self._callback,
        )

    async def _callback(self, interaction: discord.Interaction) -> None:
        channel_id = interaction.channel_id
        if channel_id is None:
            await interaction.response.send_message("Can't determine channel.", ephemeral=True)
            return
        self._bot._active_channels.add(channel_id)
        save_active_channels(self._bot._active_channels)
        await interaction.response.send_message(
            "Activated — I'll respond to every message in this channel now.", ephemeral=True
        )


class _DeactivateCommand(app_commands.Command):
    """Stop responding to all messages; require @mention again."""

    def __init__(self, bot: BeromeBot) -> None:
        self._bot = bot
        super().__init__(
            name="deactivate",
            description="Berome will only respond when @mentioned in this channel",
            callback=self._callback,
        )

    async def _callback(self, interaction: discord.Interaction) -> None:
        channel_id = interaction.channel_id
        if channel_id is None:
            await interaction.response.send_message("Can't determine channel.", ephemeral=True)
            return
        self._bot._active_channels.discard(channel_id)
        save_active_channels(self._bot._active_channels)
        await interaction.response.send_message(
            "Deactivated — I'll only respond when @mentioned.", ephemeral=True
        )


class _TeachCommand(app_commands.Command):
    """Teach the bot a persistent fact about this server."""

    def __init__(self, bot: BeromeBot) -> None:
        self._bot = bot
        super().__init__(
            name="teach",
            description="Teach Berome a persistent fact about this server or its members",
            callback=self._callback,
        )

    async def _callback(self, interaction: discord.Interaction, fact: str) -> None:
        guild_id = interaction.guild_id
        if guild_id is None:
            await interaction.response.send_message(
                "Teaching only works in a server, not in DMs.", ephemeral=True
            )
            return
        memories = add_memory(guild_id, fact)
        # Invalidate existing sessions for this guild so they pick up new memories
        await interaction.response.send_message(
            f"Got it, I'll remember that.\n*You now have {len(memories)} fact(s) stored. "
            f"Use `/clear` so I pick them up in the next message.*",
            ephemeral=True,
        )


class _ForgetCommand(app_commands.Command):
    """Remove a stored fact by index."""

    def __init__(self, bot: BeromeBot) -> None:
        self._bot = bot
        super().__init__(
            name="forget",
            description="Remove a stored memory by its number (use /memories to see the list)",
            callback=self._callback,
        )

    async def _callback(self, interaction: discord.Interaction, number: int) -> None:
        guild_id = interaction.guild_id
        if guild_id is None:
            await interaction.response.send_message(
                "This only works in a server.", ephemeral=True
            )
            return
        try:
            removed, remaining = remove_memory(guild_id, number - 1)
            await interaction.response.send_message(
                f"Forgotten: *{removed}*\n{len(remaining)} fact(s) remaining.",
                ephemeral=True,
            )
        except IndexError:
            await interaction.response.send_message(
                f"No memory at position {number}. Use `/memories` to see the list.",
                ephemeral=True,
            )


class _MemoriesCommand(app_commands.Command):
    """List all stored facts for this server."""

    def __init__(self, bot: BeromeBot) -> None:
        self._bot = bot
        super().__init__(
            name="memories",
            description="List all facts Berome has been taught about this server",
            callback=self._callback,
        )

    async def _callback(self, interaction: discord.Interaction) -> None:
        guild_id = interaction.guild_id
        if guild_id is None:
            await interaction.response.send_message(
                "This only works in a server.", ephemeral=True
            )
            return
        memories = load_memories(guild_id)
        if not memories:
            await interaction.response.send_message(
                "No memories stored yet. Use `/teach` to add some.", ephemeral=True
            )
            return
        lines = [f"{i + 1}. {m}" for i, m in enumerate(memories)]
        await interaction.response.send_message(
            "**Server memories:**\n" + "\n".join(lines), ephemeral=True
        )


# ── Role management commands ───────────────────────────────────────────────────


class _RoleGiveCommand(app_commands.Command):
    """Assign a role to a member."""

    def __init__(self, bot: BeromeBot) -> None:
        self._bot = bot
        super().__init__(
            name="role-give",
            description="Assign a role to a server member",
            callback=self._callback,
        )

    async def _callback(
        self,
        interaction: discord.Interaction,
        member: discord.Member,
        role: discord.Role,
    ) -> None:
        if interaction.guild is None:
            await interaction.response.send_message("This only works in a server.", ephemeral=True)
            return
        try:
            await member.add_roles(role, reason=f"Assigned by {interaction.user}")
            await interaction.response.send_message(
                f"Gave **{role.name}** to {member.mention}.", ephemeral=True
            )
        except discord.Forbidden:
            await interaction.response.send_message(
                "I don't have permission to assign that role — make sure my role is above it in the hierarchy.", ephemeral=True
            )
        except discord.HTTPException as exc:
            await interaction.response.send_message(f"Failed: {exc}", ephemeral=True)


class _RoleTakeCommand(app_commands.Command):
    """Remove a role from a member."""

    def __init__(self, bot: BeromeBot) -> None:
        self._bot = bot
        super().__init__(
            name="role-take",
            description="Remove a role from a server member",
            callback=self._callback,
        )

    async def _callback(
        self,
        interaction: discord.Interaction,
        member: discord.Member,
        role: discord.Role,
    ) -> None:
        if interaction.guild is None:
            await interaction.response.send_message("This only works in a server.", ephemeral=True)
            return
        try:
            await member.remove_roles(role, reason=f"Removed by {interaction.user}")
            await interaction.response.send_message(
                f"Removed **{role.name}** from {member.mention}.", ephemeral=True
            )
        except discord.Forbidden:
            await interaction.response.send_message(
                "I don't have permission to remove that role — make sure my role is above it in the hierarchy.", ephemeral=True
            )
        except discord.HTTPException as exc:
            await interaction.response.send_message(f"Failed: {exc}", ephemeral=True)


class _RoleCreateCommand(app_commands.Command):
    """Create a new role."""

    def __init__(self, bot: BeromeBot) -> None:
        self._bot = bot
        super().__init__(
            name="role-create",
            description="Create a new server role",
            callback=self._callback,
        )

    async def _callback(
        self,
        interaction: discord.Interaction,
        name: str,
        color: Optional[str] = None,
        mentionable: bool = False,
        hoist: bool = False,
    ) -> None:
        if interaction.guild is None:
            await interaction.response.send_message("This only works in a server.", ephemeral=True)
            return
        role_color = discord.Color.default()
        if color:
            try:
                role_color = discord.Color(int(color.lstrip("#"), 16))
            except ValueError:
                await interaction.response.send_message(
                    "Invalid color. Use a hex code like `#ff5733`.", ephemeral=True
                )
                return
        try:
            role = await interaction.guild.create_role(
                name=name,
                color=role_color,
                mentionable=mentionable,
                hoist=hoist,
                reason=f"Created by {interaction.user}",
            )
            await interaction.response.send_message(
                f"Created role **{role.name}** (ID: `{role.id}`).", ephemeral=True
            )
        except discord.Forbidden:
            await interaction.response.send_message(
                "I don't have permission to create roles.", ephemeral=True
            )
        except discord.HTTPException as exc:
            await interaction.response.send_message(f"Failed: {exc}", ephemeral=True)


class _RoleDeleteCommand(app_commands.Command):
    """Delete an existing role."""

    def __init__(self, bot: BeromeBot) -> None:
        self._bot = bot
        super().__init__(
            name="role-delete",
            description="Delete a server role",
            callback=self._callback,
        )

    async def _callback(
        self,
        interaction: discord.Interaction,
        role: discord.Role,
    ) -> None:
        if interaction.guild is None:
            await interaction.response.send_message("This only works in a server.", ephemeral=True)
            return
        name = role.name
        try:
            await role.delete(reason=f"Deleted by {interaction.user}")
            await interaction.response.send_message(f"Deleted role **{name}**.", ephemeral=True)
        except discord.Forbidden:
            await interaction.response.send_message(
                "I don't have permission to delete that role — make sure my role is above it in the hierarchy.", ephemeral=True
            )
        except discord.HTTPException as exc:
            await interaction.response.send_message(f"Failed: {exc}", ephemeral=True)


class _RoleListCommand(app_commands.Command):
    """List all roles in the server."""

    def __init__(self, bot: BeromeBot) -> None:
        self._bot = bot
        super().__init__(
            name="role-list",
            description="List all roles in this server",
            callback=self._callback,
        )

    async def _callback(self, interaction: discord.Interaction) -> None:
        if interaction.guild is None:
            await interaction.response.send_message("This only works in a server.", ephemeral=True)
            return
        roles = [r for r in reversed(interaction.guild.roles) if r.name != "@everyone"]
        if not roles:
            await interaction.response.send_message("No roles found.", ephemeral=True)
            return
        lines = [
            f"`{r.id}` **{r.name}** — {len(r.members)} member(s)"
            + (" *(mentionable)*" if r.mentionable else "")
            for r in roles
        ]
        text = "**Server roles:**\n" + "\n".join(lines)
        # Split if too long
        if len(text) > DISCORD_MAX_LENGTH:
            text = text[: DISCORD_MAX_LENGTH - 3] + "..."
        await interaction.response.send_message(text, ephemeral=True)


class _DocumentCommand(app_commands.Command):
    """Generate and send an essay or research document as a file."""

    def __init__(self, bot: BeromeBot) -> None:
        self._bot = bot
        super().__init__(
            name="document",
            description="Ask Berome to write an essay/document and send it as a file",
            callback=self._callback,
        )

    async def _callback(
        self,
        interaction: discord.Interaction,
        topic: str,
        format: Optional[str] = "md",
    ) -> None:
        """
        Parameters
        ----------
        topic  : What to write about (e.g. "the causes of WW1")
        format : Output format — md (default), html, or txt
        """
        fmt = (format or "md").lower().lstrip(".")
        if fmt not in {"md", "html", "txt"}:
            await interaction.response.send_message(
                "Format must be `md`, `html`, or `txt`.", ephemeral=True
            )
            return

        await interaction.response.send_message(
            f"Writing a document on **{topic}** ({fmt.upper()})...", ephemeral=False
        )

        channel = interaction.channel
        if channel is None:
            return

        session = await self._bot._get_session(channel)
        if session is None:
            await channel.send("LLM provider unavailable.")
            return

        filename = "_".join(topic.lower().split()[:6])[:40] + f".{fmt}"
        prompt = (
            f"[System]: Write a thorough, well-structured essay or research document about: {topic}. "
            f"Save it using write_file with path '{filename}'. "
            f"Format it as {'Markdown' if fmt == 'md' else fmt.upper()}."
        )

        written_docs: list[Path] = []
        _pending: list[str] = []

        _DOC_TOOL_STATUS: dict[str, str] = {
            "web_search": "🔍 Researching: **{query}**",
            "image_search": "🖼️ Finding images: **{query}**",
            "read_file": "📖 Reading: `{path}`",
            "write_file": "📝 Writing document: `{path}`",
        }

        async def on_tool_call(name: str, args: dict) -> None:
            if name == "write_file":
                path = args.get("path", "")
                if path:
                    _pending.append(path)
            template = _DOC_TOOL_STATUS.get(name)
            if template:
                try:
                    status = template.format_map({k: str(v)[:80] for k, v in args.items()})
                    await channel.send(status)
                except (discord.HTTPException, KeyError):
                    pass

        async def on_tool_result(result) -> None:
            if result.tool_name == "write_file" and not result.error and _pending:
                p = Path(_pending.pop(0))
                if p.exists():
                    written_docs.append(p)

        async def no_confirm(command: str) -> bool:
            return False  # No shell commands needed for document generation

        response_parts: list[str] = []
        try:
            async with channel.typing():
                async for chunk in session.agentic_stream(
                    prompt, on_tool_call, on_tool_result, no_confirm
                ):
                    response_parts.append(chunk)
        except Exception as exc:
            logger.exception("Error during document generation")
            await channel.send(f"Error: {str(exc)[:300]}")
            return

        if written_docs:
            await _send_documents(written_docs, channel)
        else:
            # Fallback: send the response as a text file if no file was written
            full = "".join(response_parts).strip()
            if full:
                ext = fmt
                fb_filename = "_".join(topic.lower().split()[:6])[:40] + f".{ext}"
                await channel.send(
                    files=[discord.File(io.BytesIO(full.encode()), filename=fb_filename)]
                )


class _SearchCommand(app_commands.Command):
    """Search server message history for a word or phrase."""

    def __init__(self, bot: BeromeBot) -> None:
        self._bot = bot
        super().__init__(
            name="search",
            description="Search server message history for a word or phrase",
            callback=self._callback,
        )

    async def _callback(
        self,
        interaction: discord.Interaction,
        term: str,
        channel: Optional[discord.TextChannel] = None,
    ) -> None:
        """
        Parameters
        ----------
        term    : Word or phrase to search for (case-insensitive)
        channel : Limit search to a specific channel (searches all by default)
        """
        if interaction.guild is None:
            await interaction.response.send_message("This only works in a server.", ephemeral=True)
            return

        await interaction.response.send_message(
            f"🔍 Searching for **{term}**{f' in #{channel.name}' if channel else ' across all channels'}...",
            ephemeral=False,
        )

        search_channels: list[discord.TextChannel] = (
            [channel] if channel else list(interaction.guild.text_channels)
        )

        term_lower = term.lower()
        total_count = 0
        channel_counts: dict[str, int] = {}
        samples: list[str] = []  # up to 5 example messages

        for ch in search_channels:
            try:
                ch_count = 0
                async for msg in ch.history(limit=2000, oldest_first=False):
                    if term_lower in msg.content.lower():
                        ch_count += 1
                        if len(samples) < 5:
                            ts = msg.created_at.strftime("%Y-%m-%d %H:%M")
                            snippet = msg.content.replace("\n", " ")[:120]
                            samples.append(
                                f"> **#{ch.name}** · {msg.author.display_name} · {ts}\n> {snippet}"
                            )
                if ch_count:
                    channel_counts[ch.name] = ch_count
                    total_count += ch_count
            except (discord.Forbidden, discord.HTTPException):
                pass

        if total_count == 0:
            await interaction.followup.send(
                f"No messages found containing **{term}** (searched last 2000 messages per channel)."
            )
            return

        breakdown = "\n".join(
            f"  #{name}: **{count}**" for name, count in sorted(channel_counts.items(), key=lambda x: -x[1])
        )
        sample_text = "\n\n".join(samples)

        result = (
            f"**Search results for \"{term}\"**\n"
            f"Total: **{total_count}** mention(s) across {len(channel_counts)} channel(s)\n\n"
            f"**By channel:**\n{breakdown}"
        )
        if samples:
            result += f"\n\n**Examples:**\n{sample_text}"

        # Split if too long
        for chunk in _split_message(result):
            await interaction.followup.send(chunk)


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
            "- Attach images to any message — I can analyse them\n"
            "- In DMs, just type freely\n\n"
            "**Slash Commands:**\n"
            "`/activate` — respond to ALL messages in this channel (no @mention)\n"
            "`/deactivate` — go back to @mention-only mode\n"
            "`/teach <fact>` — teach me something to remember about this server\n"
            "`/memories` — list everything I've been taught\n"
            "`/forget <number>` — remove a memory by its number\n"
            "`/clear` — clear conversation history for this channel\n"
            "`/status` — show session info and token usage\n"
            "`/provider [name]` — show or switch LLM provider (`anthropic` / `ollama`)\n"
            "`/berome-help` — show this help\n\n"
            "**Role Management**\n"
            "`/role-give <member> <role>` — assign a role to a member\n"
            "`/role-take <member> <role>` — remove a role from a member\n"
            "`/role-create <name> [color] [mentionable] [hoist]` — create a new role\n"
            "`/role-delete <role>` — delete a role\n"
            "`/role-list` — list all server roles\n\n"
            "**Documents:**\n"
            "`/document <topic> [format]` — write an essay/research doc and send it as a file (md/html/txt)\n\n"
            "**Search:**\n"
            "`/search <term> [channel]` — count how many times a word/phrase appears in server history\n\n"
            "**Tools:**\n"
            "📖 read_file  📝 write_file  📁 list_directory  📂 create_directory\n"
            "⚙️ run_command *(requires confirmation)*  🗑️ delete_file  🔍 web_search"
        )
        await interaction.response.send_message(help_text, ephemeral=True)


# ── Utility functions ──────────────────────────────────────────────────────────


async def _send_response(content: str, channel: discord.abc.Messageable) -> None:
    """Send the full response, splitting across multiple messages if over 2000 chars."""
    for chunk in _split_message(content):
        try:
            await channel.send(chunk)
        except discord.HTTPException as exc:
            logger.warning("Failed to send response chunk: %s", exc)


async def _send_documents(docs: list[Path], channel: discord.abc.Messageable) -> None:
    """Attach written document files to a Discord message.

    Supports .md, .html, .txt, and .pdf directly.
    Optionally converts .md to .html if the markdown package is available.
    """
    files: list[discord.File] = []
    for doc in docs:
        try:
            data = doc.read_bytes()
            files.append(discord.File(io.BytesIO(data), filename=doc.name))

            # If it's a .md file, also offer an HTML version
            if doc.suffix.lower() == ".md":
                try:
                    import markdown as md_lib
                    text = doc.read_text(encoding="utf-8")
                    html_content = md_lib.markdown(text, extensions=["tables", "fenced_code"])
                    html_full = (
                        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                        "<style>body{font-family:sans-serif;max-width:800px;margin:auto;padding:2em}"
                        "pre{background:#f4f4f4;padding:1em;border-radius:4px;overflow:auto}"
                        "code{background:#f4f4f4;padding:.2em .4em;border-radius:3px}"
                        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.5em}"
                        "</style></head><body>"
                        f"{html_content}</body></html>"
                    )
                    html_name = doc.stem + ".html"
                    files.append(discord.File(io.BytesIO(html_full.encode()), filename=html_name))
                except ImportError:
                    pass
        except Exception as exc:
            logger.warning("Failed to attach document %s: %s", doc, exc)

    if not files:
        return
    try:
        # Discord allows up to 10 files per message
        for i in range(0, len(files), 10):
            await channel.send(files=files[i:i + 10])
    except discord.HTTPException as exc:
        logger.warning("Failed to send document attachments: %s", exc)


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

        split_pos = text.rfind("\n", 0, limit)
        if split_pos == -1:
            split_pos = limit

        chunk = text[:split_pos]
        text = text[split_pos:].lstrip("\n")

        fence_count = chunk.count("```")
        if fence_count % 2 == 1:
            chunk = chunk + "\n```"
            text = "```\n" + text

        chunks.append(chunk)

    return chunks
