"""Orchestrator: the single entry point behind /ask.

Uses the agents-as-tools pattern - each specialist is exposed to the
orchestrator as a tool, and the orchestrator routes the user's request to the
right one and composes the final answer. All agents share one Bedrock model.
"""

from __future__ import annotations

from strands import Agent

from .curator_agent import build_curator_agent
from .ingestion_agent import build_ingestion_agent
from .search_agent import build_search_agent
from .shared import build_bedrock_model

SYSTEM_PROMPT = """You are the assistant for a personal image-search app. Route the
user's request to the right specialist tool and give a concise, friendly reply.

- Finding/showing images (by description or "more like this") -> search_specialist
- Questions about the collection (how many, how big, what kinds) -> curator_specialist
- Adding/indexing new images from a folder or zip path -> ingestion_specialist

Prefer a single specialist call. Don't fabricate results the tools didn't return.
"""


def build_orchestrator(model=None) -> Agent:
    model = model or build_bedrock_model()
    search_agent = build_search_agent(model)
    curator_agent = build_curator_agent(model)
    ingestion_agent = build_ingestion_agent(model)

    return Agent(
        name="orchestrator",
        system_prompt=SYSTEM_PROMPT,
        model=model,
        tools=[
            search_agent.as_tool(
                name="search_specialist",
                description="Find images from a natural-language description or a reference image.",
            ),
            curator_agent.as_tool(
                name="curator_specialist",
                description="Answer questions about the collection (size, counts, what kinds of images).",
            ),
            ingestion_agent.as_tool(
                name="ingestion_specialist",
                description="Index new images from a local folder or .zip path.",
            ),
        ],
    )
