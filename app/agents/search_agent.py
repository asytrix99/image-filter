"""Search specialist: turns fuzzy natural language into good vector searches."""

from __future__ import annotations

from strands import Agent

from .shared import build_bedrock_model, find_similar_to, search_images

SYSTEM_PROMPT = """You are a visual search specialist for a personal image collection.

Your job: understand what the user is really looking for, then call the search
tools to find it.
- Rewrite vague requests into clear visual concepts before searching
  (e.g. "pics from our beach trip" -> search "beach"; "the pups" -> "dog").
- You may search more than once and combine what you find.
- Use find_similar_to when the user references a specific image id.
Reply with a short, friendly summary naming the top few matching files. Do not
invent files that the tools did not return.
"""


def build_search_agent(model=None) -> Agent:
    return Agent(
        name="search_agent",
        description="Finds images from a natural-language description or a reference image.",
        system_prompt=SYSTEM_PROMPT,
        model=model or build_bedrock_model(),
        tools=[search_images, find_similar_to],
    )
