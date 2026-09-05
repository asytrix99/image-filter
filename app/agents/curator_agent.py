"""Curator specialist: answers analytical questions about the collection."""

from __future__ import annotations

from strands import Agent

from .shared import build_bedrock_model, count_matching, describe_collection

SYSTEM_PROMPT = """You are a collection curator for a personal image library.

You answer questions *about* the collection rather than just retrieving images:
- "how many photos do I have" -> describe_collection
- "how many dogs / sunsets / people" -> count_matching with a sensible concept
Be clear that counts are similarity-based estimates, not exact object detection.
Keep answers concise.
"""


def build_curator_agent(model=None) -> Agent:
    return Agent(
        name="curator_agent",
        description="Answers questions about the collection (size, how many of X, what kinds of images).",
        system_prompt=SYSTEM_PROMPT,
        model=model or build_bedrock_model(),
        tools=[describe_collection, count_matching],
    )
