"""Phase 1 CLI: build the search index from a zip or folder of images.

Usage (from image-filter/):
    ./.venv/Scripts/python.exe -m scripts.build_index path/to/images.zip
    ./.venv/Scripts/python.exe -m scripts.build_index path/to/folder/

This is the glue that ties Modules 1-3 together: Source -> prepare -> embed ->
store. The Phase 3 ingestion agent will later orchestrate this same sequence.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from app import config, embed, ingest
from app.sources import FolderSource, ImageItem, ZipSource
from app.store import QdrantStore


def _source(path: Path):
    if path.is_dir():
        return FolderSource(path)
    if path.suffix.lower() == ".zip":
        return ZipSource(path)
    raise SystemExit(f"Expected a .zip file or a folder, got: {path}")


def build_index(path: Path, batch_size: int = config.EMBED_BATCH_SIZE) -> int:
    config.ensure_dirs()
    src = _source(path)
    store = QdrantStore()

    total = src.count()
    print(f"Found {total} candidate images in {path.name}")

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
        print(f"  indexed {indexed}/{total} ...")
        batch_items.clear()
        batch_payloads.clear()

    t0 = time.time()
    for item in src.iter_images():
        payload = ingest.prepare(item)          # validate + thumbnail + persist
        if payload is None:
            skipped += 1
            continue
        batch_items.append(item)
        batch_payloads.append(payload)
        if len(batch_items) >= batch_size:
            flush()
    flush()

    dt = time.time() - t0
    print(f"\nDone: indexed {indexed}, skipped {skipped} (bad/corrupt), "
          f"in {dt:.1f}s. Collection now holds {store.count()} images.")
    return indexed


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m scripts.build_index <zip-or-folder>")
    build_index(Path(sys.argv[1]))
