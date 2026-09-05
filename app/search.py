"""Module 4 - search.

Thin layer that turns a query (text OR an example image) into a vector and asks
the VectorStore for the nearest images. Text-search and image-search share the
same code path because CLIP puts both in the same space.
"""

from __future__ import annotations

import numpy as np

from . import embed
from .store import QdrantStore, VectorStore


def _store() -> VectorStore:
    return QdrantStore()


def search(query: str, top_k: int = 50, query_filter=None,
           store: VectorStore | None = None) -> list[tuple[str, float, dict]]:
    """Text search: embed the query, return nearest (image_id, score, payload)."""
    store = store or _store()
    vec = embed.embed_text(query)[0]
    return store.search(vec, top_k=top_k, query_filter=query_filter)


def search_by_image(image_bytes: bytes, top_k: int = 50, query_filter=None,
                    store: VectorStore | None = None) -> list[tuple[str, float, dict]]:
    """'Find more like this': embed the example image, return nearest images.

    Same store.search call as text search - only the encoder differs.
    """
    store = store or _store()
    vec = embed.embed_images([image_bytes])[0]
    return store.search(vec, top_k=top_k, query_filter=query_filter)


def search_combined(image_bytes: bytes, text: str, top_k: int = 50,
                    image_weight: float = 0.5, query_filter=None,
                    store: VectorStore | None = None) -> list[tuple[str, float, dict]]:
    """Blend an example image and a text query ('more like this, but outdoors').

    Because both vectors live in the same space, a weighted average is a valid
    query vector; we re-normalize so cosine ranking stays meaningful.
    """
    store = store or _store()
    img_vec = embed.embed_images([image_bytes])[0]
    txt_vec = embed.embed_text(text)[0]
    vec = image_weight * img_vec + (1.0 - image_weight) * txt_vec
    vec = vec / np.linalg.norm(vec)
    return store.search(vec, top_k=top_k, query_filter=query_filter)
