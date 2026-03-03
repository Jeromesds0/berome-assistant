"""Shared Rich styles and theme constants."""

from rich.style import Style
from rich.theme import Theme

BEROME_THEME = Theme(
    {
        "berome.header": "bold cyan",
        "berome.user": "bold green",
        "berome.assistant": "bold blue",
        "berome.system": "dim yellow",
        "berome.error": "bold red",
        "berome.success": "bold green",
        "berome.warning": "bold yellow",
        "berome.muted": "dim white",
        "berome.agent": "bold magenta",
        "berome.github": "bold white on dark_green",
        "berome.code": "bold bright_cyan",
    }
)

BANNER = r"""
[bold cyan]
 ██████╗ ███████╗██████╗  ██████╗ ███╗   ███╗███████╗
 ██╔══██╗██╔════╝██╔══██╗██╔═══██╗████╗ ████║██╔════╝
 ██████╔╝█████╗  ██████╔╝██║   ██║██╔████╔██║█████╗
 ██╔══██╗██╔══╝  ██╔══██╗██║   ██║██║╚██╔╝██║██╔══╝
 ██████╔╝███████╗██║  ██║╚██████╔╝██║ ╚═╝ ██║███████╗
 ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝
[/bold cyan]
[dim]  Your AI Personal Assistant  •  type /help for commands[/dim]
"""
