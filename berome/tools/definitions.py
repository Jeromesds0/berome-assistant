"""
Tool definitions in a provider-neutral format.

Each entry uses ``parameters`` (JSON Schema object) so that:
- Anthropic providers remap ``parameters`` → ``input_schema``
- Ollama/OpenAI providers wrap as ``{"type": "function", "function": {...}}``
"""

from __future__ import annotations

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "write_file",
        "description": (
            "Write text content to a file on the local filesystem, creating "
            "parent directories and the file if they don't exist."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative or absolute file path to write.",
                },
                "content": {
                    "type": "string",
                    "description": "Text content to write to the file.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read the text content of a file from the local filesystem and "
            "return it as a string."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative or absolute file path to read.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_directory",
        "description": (
            "List the files and subdirectories inside a directory, returning "
            "names and types."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list (defaults to '.').",
                },
            },
        },
    },
    {
        "name": "create_directory",
        "description": (
            "Create a directory (and any missing parent directories) on the "
            "local filesystem."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to create.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "run_command",
        "description": (
            "Run a shell command and return its combined stdout+stderr output. "
            "The user will be shown the command and asked to confirm before "
            "execution. Use for installing packages, running scripts, git "
            "operations, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute.",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "image_search",
        "description": (
            "Search the web for images and return direct image URLs. "
            "Use this when asked to find, send, or share a meme, photo, or any image. "
            "Returns real URLs that Discord will auto-embed. Never guess or fabricate URLs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The image search query, e.g. 'gigachad meme' or 'cat funny'.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of image URLs to return (default 1, max 5).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web using DuckDuckGo and return the top results. "
            "Use this to look up current events, pop culture, memes, recent news, "
            "documentation, or anything that requires up-to-date information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, max 10).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "delete_file",
        "description": (
            "Delete a file (not a directory) from the local filesystem."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to delete.",
                },
            },
            "required": ["path"],
        },
    },
]
