"""Module 1a - image sources.

A `Source` decouples *where images come from* from the rest of the pipeline
(embed -> store -> search). v1 ships `ZipSource` and `FolderSource`. A future
`McpSource` can pull images from a cloud store via an MCP server without changing
anything downstream.

Each source yields `ImageItem`s: a stable id, the raw image bytes, and a small
metadata dict. Yielding *bytes* (not paths) is what lets a future cloud source
stream remote objects that never touch local disk.
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol

from . import config


@dataclass
class ImageItem:
    """One image flowing through the pipeline."""

    id: str                      # stable, content-addressed id (see _make_id)
    data: bytes                  # raw encoded image bytes
    filename: str                # original name, for display
    meta: dict = field(default_factory=dict)


def _make_id(data: bytes) -> str:
    """Content-addressed id: sha1 of the bytes.

    Using the content hash (not a counter) means the same image gets the same id
    across re-ingests, which gives us free de-duplication in the vector store.
    """
    return hashlib.sha1(data).hexdigest()


def _is_image_name(name: str) -> bool:
    return Path(name).suffix.lower() in config.IMAGE_EXTS


class Source(Protocol):
    """Anything the pipeline can ingest from."""

    def count(self) -> int:
        """Approximate number of images (for progress reporting)."""
        ...

    def iter_images(self) -> Iterator[ImageItem]:
        """Yield images one at a time."""
        ...


class ZipSource:
    """Reads images out of a .zip file, safely.

    Guards against:
      - zip-slip / path traversal (entries like ``../../evil.jpg``)
      - too many files or too large a total (config.MAX_IMAGES / MAX_ZIP_BYTES)
    """

    def __init__(self, zip_path: str | Path):
        self.zip_path = Path(zip_path)
        if not self.zip_path.is_file():
            raise FileNotFoundError(self.zip_path)

    def _safe_members(self, zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
        members: list[zipfile.ZipInfo] = []
        total = 0
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            # zip-slip guard: reject absolute paths and any ".." traversal.
            if name.startswith(("/", "\\")) or ".." in Path(name).parts:
                continue
            if not _is_image_name(name):
                continue
            total += info.file_size
            if total > config.MAX_ZIP_BYTES:
                raise ValueError(
                    f"zip exceeds MAX_ZIP_BYTES ({config.MAX_ZIP_BYTES} bytes)"
                )
            members.append(info)
            if len(members) > config.MAX_IMAGES:
                raise ValueError(f"zip exceeds MAX_IMAGES ({config.MAX_IMAGES})")
        return members

    def count(self) -> int:
        with zipfile.ZipFile(self.zip_path) as zf:
            return len(self._safe_members(zf))

    def iter_images(self) -> Iterator[ImageItem]:
        with zipfile.ZipFile(self.zip_path) as zf:
            for info in self._safe_members(zf):
                data = zf.read(info)
                yield ImageItem(
                    id=_make_id(data),
                    data=data,
                    filename=Path(info.filename).name,
                    meta={"source": "zip", "zip": self.zip_path.name},
                )


class FolderSource:
    """Reads images from a local directory tree (handy for the CLI demo)."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise NotADirectoryError(self.folder)

    def _files(self) -> list[Path]:
        return [p for p in sorted(self.folder.rglob("*")) if p.is_file() and _is_image_name(p.name)]

    def count(self) -> int:
        return len(self._files())

    def iter_images(self) -> Iterator[ImageItem]:
        for p in self._files():
            data = p.read_bytes()
            yield ImageItem(
                id=_make_id(data),
                data=data,
                filename=p.name,
                meta={"source": "folder", "path": str(p)},
            )
