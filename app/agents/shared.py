"""Shared plumbing for the agents: the Bedrock model factory, the @tool
wrappers around the Phase 1-2 pipeline, and a small mechanism to capture the
structured image results a tool produced during a turn (so the web UI can show
thumbnails, not just the agent's text answer).

Design rule: tools do the real work (vector search, indexing) and return short
text the LLM can reason over. The model never touches raw pixels or loops per
image - it only decides *what* to search / index.
"""

from __future__ import annotations

import contextvars
from pathlib import Path

from strands import tool
from strands.models import BedrockModel

from .. import config, indexer, search
from ..sources import FolderSource, ZipSource
from ..store import get_store

# --- result capture ---------------------------------------------------------
# A tool runs inside agent(...) on the same context, so a ContextVar lets the
# /ask endpoint read whatever images the search tool found during the turn.
_results: contextvars.ContextVar[list | None] = contextvars.ContextVar("results", default=None)


def reset_results() -> None:
    _results.set([])


def take_results() -> list:
    return _results.get() or []


def _fmt(hits) -> list[dict]:
    return [
        {
            "image_id": iid,
            "score": round(float(score), 4),
            "filename": (payload or {}).get("filename"),
            "thumb_url": f"/thumb/{iid}",
            "image_url": f"/image/{iid}",
        }
        for iid, score, payload in hits
    ]


def _capture(hits) -> None:
    lst = _results.get()
    if lst is not None:
        lst.extend(_fmt(hits))


# --- model factory ----------------------------------------------------------
def build_bedrock_model(temperature: float = 0.3) -> BedrockModel:
    """One place to construct the Bedrock model all agents share."""
    return BedrockModel(
        model_id=config.BEDROCK_MODEL_ID,
        region_name=config.BEDROCK_REGION,
        temperature=temperature,
    )


# --- tools: search ----------------------------------------------------------
@tool
def search_images(query: str, k: int = 20) -> str:
    """Search the image collection for a text description and return the top matches.

    Args:
        query: what to look for, e.g. "a dog on the beach", "birthday cake".
        k: how many results to return (default 20).
    """
    hits = search.search(query, top_k=k, store=get_store())
    _capture(hits)
    if not hits:
        return "No images matched."
    return "Top matches:\n" + "\n".join(
        f"{i+1}. {(p or {}).get('filename', iid)} (score {s:.3f})"
        for i, (iid, s, p) in enumerate(hits)
    )


@tool
def find_similar_to(image_id: str, k: int = 20) -> str:
    """Find images visually similar to one already in the collection (by its id)."""
    matches = list(config.IMAGES_DIR.glob(f"{image_id}.*"))
    if not matches:
        return f"No stored image with id {image_id}."
    hits = search.search_by_image(matches[0].read_bytes(), top_k=k, store=get_store())
    _capture(hits)
    return "Similar images:\n" + "\n".join(
        f"{i+1}. {(p or {}).get('filename', iid)} (score {s:.3f})"
        for i, (iid, s, p) in enumerate(hits)
    )


# --- tools: curation / stats ------------------------------------------------
@tool
def describe_collection() -> str:
    """Report how many images are indexed (use for 'how big is my collection')."""
    return f"The collection currently holds {get_store().count()} indexed images."


@tool
def count_matching(query: str, min_score: float = 0.22, k: int = 100) -> str:
    """Estimate how many images match a concept (e.g. 'how many dogs').

    Counts results whose similarity score exceeds min_score. This is an estimate:
    CLIP ranks by similarity, it does not detect/verify objects.
    """
    hits = search.search(query, top_k=k, store=get_store())
    strong = [(iid, s, p) for iid, s, p in hits if s >= min_score]
    _capture(strong)
    return (f"~{len(strong)} images look like '{query}' "
            f"(similarity >= {min_score}). This is an estimate, not exact detection.")


# --- tools: ingestion -------------------------------------------------------
@tool
def index_path(path: str) -> str:
    """Index images into the collection from a local folder or .zip file path."""
    p = Path(path)
    if p.is_dir():
        src = FolderSource(p)
    elif p.suffix.lower() == ".zip" and p.is_file():
        src = ZipSource(p)
    else:
        return f"Path is not a folder or .zip: {path}"
    result = indexer.index_source(src, store=get_store())
    return (f"Indexed {result['indexed']} images, skipped {result['skipped']} "
            f"(corrupt/unreadable). Collection now holds {get_store().count()}.")
