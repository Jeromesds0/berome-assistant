"""Code agent – uses the LLM to write, review, and explain code."""

from __future__ import annotations

from berome.agents.base import Agent, AgentTask
from berome.providers.base import LLMMessage, LLMProvider


CODE_SYSTEM = """\
You are an expert software engineer. When asked to write code:
- Produce clean, idiomatic code with brief comments for non-obvious logic.
- Always specify the language in fenced code blocks.
- If reviewing code, identify bugs, security issues, and improvements.
- If explaining code, be concise and clear.\
"""


class CodeAgent(Agent):
    agent_type = "code"
    description = "Writes, reviews, and explains code using the LLM"

    def __init__(self, provider: LLMProvider) -> None:
        super().__init__(provider)

    async def run(self, task: AgentTask) -> str:
        """
        payload keys:
          mode    – "write" | "review" | "explain" | "refactor"
          prompt  – the user's request or code snippet
          language – optional language hint
        """
        mode: str = task.payload.get("mode", "write")
        prompt: str = task.payload.get("prompt", "")
        language: str = task.payload.get("language", "")

        prefixes = {
            "write": f"Write the following code{' in ' + language if language else ''}:\n\n",
            "review": "Review this code and identify issues:\n\n",
            "explain": "Explain what this code does:\n\n",
            "refactor": "Refactor this code for clarity and performance:\n\n",
        }
        user_content = prefixes.get(mode, "") + prompt

        response = await self._llm.chat(
            messages=[LLMMessage(role="user", content=user_content)],
            system=CODE_SYSTEM,
        )
        return response.content
