"""Core indexing loop, shared by the CLI (scripts/build_index.py) and the web
API (app/api.py).

Ties Modules 1-3 together: Source -> ingest.prepare -> embed -> store.upsert,
in batches, with an optional progress callback so a caller (CLI print / API job
status) can report how far along it is.
"""

from __future__ import annotations

from typing import Callable

from . import config, embed, ingest
from .sources import ImageItem, Source
from .store import QdrantStore, VectorStore

# on_progress(indexed, skipped, total)
ProgressCb = Callable[[int, int, int], None]


def index_source(src: Source, store: VectorStore | None = None,
                 batch_size: int = config.EMBED_BATCH_SIZE,
                 on_progress: ProgressCb | None = None) -> dict:
    """Index every image from `src`. Returns {indexed, skipped, total}."""
    config.ensure_dirs()
    store = store or QdrantStore()
    total = src.count()

    indexed = 0
    skipped = 0
    batch_items: list[ImageItem] = []
    batch_payloads: list[dict] = []

    def flush() -> None:
        nonlocal indexed
        if not batch_items:
            return
        vectors = embed.embed_images([it.data for it in batch_items], batch_size)
        store.upsert(
            ids=[it.id for it in batch_items],
            vectors=vectors,
            payloads=batch_payloads,
        )
        indexed += len(batch_items)
        if on_progress:
            on_progress(indexed, skipped, total)
        batch_items.clear()
        batch_payloads.clear()

    for item in src.iter_images():
        payload = ingest.prepare(item)      # validate + thumbnail + persist
        if payload is None:
            skipped += 1
            continue
        batch_items.append(item)
        batch_payloads.append(payload)
        if len(batch_items) >= batch_size:
            flush()
    flush()

    return {"indexed": indexed, "skipped": skipped, "total": total}
