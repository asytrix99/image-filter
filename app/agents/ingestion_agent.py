"""Ingestion specialist: indexes new images and reports on failures.

For now it wraps index_path (which validates, retries truncated files via the
ingest layer, and quarantines the unrecoverable). This is where the future
McpSource / smarter batching decisions will live.
"""

from __future__ import annotations

from strands import Agent

from .shared import build_bedrock_model, describe_collection, index_path

SYSTEM_PROMPT = """You manage ingestion for a personal image collection.

When asked to add/index images from a folder or zip path, call index_path and
then report how many were indexed vs skipped (corrupt/unreadable). If the user
asks to confirm the result, you can call describe_collection. Be concise.
"""


def build_ingestion_agent(model=None) -> Agent:
    return Agent(
        name="ingestion_agent",
        description="Indexes new images from a folder or zip path and reports results.",
        system_prompt=SYSTEM_PROMPT,
        model=model or build_bedrock_model(),
        tools=[index_path, describe_collection],
    )
