"""Research agent – uses the LLM to research topics and summarise findings."""

from __future__ import annotations

from berome.agents.base import Agent, AgentTask
from berome.providers.base import LLMMessage, LLMProvider


RESEARCH_SYSTEM = """\
You are a thorough research assistant. When given a topic:
- Summarise the key facts, concepts, and current state.
- Highlight trade-offs, pros/cons, and edge cases.
- Where relevant, suggest resources or next steps.
- Be concise; use bullet points and headers.\
"""


class ResearchAgent(Agent):
    agent_type = "research"
    description = "Researches topics and returns structured summaries"

    def __init__(self, provider: LLMProvider) -> None:
        super().__init__(provider)

    async def run(self, task: AgentTask) -> str:
        topic: str = task.payload.get("topic", task.payload.get("prompt", ""))
        depth: str = task.payload.get("depth", "medium")  # brief | medium | deep

        depth_instructions = {
            "brief": "Give a 3-5 bullet summary.",
            "medium": "Give a structured overview with 2-3 paragraphs or sections.",
            "deep": "Give a comprehensive deep-dive with sections, examples, and trade-offs.",
        }

        user_content = (
            f"Research this topic: {topic}\n\n"
            f"{depth_instructions.get(depth, depth_instructions['medium'])}"
        )

        response = await self._llm.chat(
            messages=[LLMMessage(role="user", content=user_content)],
            system=RESEARCH_SYSTEM,
        )
        return response.content
