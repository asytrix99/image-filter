"""Tests for Modules 3+4 (store + search) using in-memory Qdrant + real CLIP.

These exercise the real embedding model, so the first run downloads the CLIP
weights (cached afterwards). We assert on self-similarity / ranking structure,
which holds regardless of the (synthetic) image content.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image
from qdrant_client import QdrantClient

from app import embed, search
from app.store import QdrantStore


def _img_bytes(color, size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def store():
    return QdrantStore(collection="test", client=QdrantClient(location=":memory:"))


def test_upsert_count_and_get(store):
    data = _img_bytes((255, 0, 0))
    vecs = embed.embed_images([data])
    store.upsert(ids=["img-red"], vectors=vecs, payloads=[{"filename": "red.png"}])
    assert store.count() == 1
    assert store.get("img-red")["filename"] == "red.png"


def test_search_by_image_ranks_self_first(store):
    reds = _img_bytes((220, 20, 20))
    blue = _img_bytes((20, 20, 220))
    green = _img_bytes((20, 220, 20))
    items = {"red": reds, "blue": blue, "green": green}
    vecs = embed.embed_images(list(items.values()))
    store.upsert(ids=list(items), vectors=vecs, payloads=[{"filename": k} for k in items])

    results = search.search_by_image(reds, top_k=3, store=store)
    # The query image is in the index, so it should rank itself first.
    assert results[0][2]["filename"] == "red"
    assert len(results) == 3


def test_embeddings_are_unit_normalized():
    v = embed.embed_text("a photo of a dog")
    assert v.shape == (1, 512)
    assert np.isclose(np.linalg.norm(v[0]), 1.0, atol=1e-4)
