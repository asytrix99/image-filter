"""Phase 1 CLI: build the search index from a zip or folder of images.

Usage (from image-filter/):
    ./.venv/Scripts/python.exe -m scripts.build_index path/to/images.zip
    ./.venv/Scripts/python.exe -m scripts.build_index path/to/folder/

Thin wrapper over app.indexer.index_source (shared with the web API).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from app import config, indexer
from app.sources import FolderSource, ZipSource
from app.store import QdrantStore


def _source(path: Path):
    if path.is_dir():
        return FolderSource(path)
    if path.suffix.lower() == ".zip":
        return ZipSource(path)
    raise SystemExit(f"Expected a .zip file or a folder, got: {path}")


def build_index(path: Path) -> int:
    src = _source(path)
    store = QdrantStore()
    print(f"Found {src.count()} candidate images in {path.name}")

    t0 = time.time()
    result = indexer.index_source(
        src, store=store,
        on_progress=lambda i, s, t: print(f"  indexed {i}/{t} ..."),
    )
    dt = time.time() - t0
    print(f"\nDone: indexed {result['indexed']}, skipped {result['skipped']} "
          f"(bad/corrupt), in {dt:.1f}s. Collection now holds {store.count()} images.")
    return result["indexed"]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m scripts.build_index <zip-or-folder>")
    build_index(Path(sys.argv[1]))
