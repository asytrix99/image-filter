"""Module 1b - validation + thumbnails.

Turns raw `ImageItem` bytes into something safe to embed and display:
  - validate_image: confirm the bytes actually decode as an image
  - make_thumbnail: write a small thumbnail for the results grid
  - save_original:  persist the full-size image so the UI can serve it

These are plain functions (no LLM) - the Phase 3 ingestion *agent* will call
them and decide what to do with failures (retry / quarantine).
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageFile, UnidentifiedImageError

from . import config
from .sources import ImageItem

# Pillow raises on truncated files by default; allow partial decode so a
# slightly-truncated JPEG is still usable rather than lost.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def validate_image(data: bytes) -> bool:
    """Return True if `data` decodes as an image, False otherwise.

    Uses Image.verify(), which checks the file is not broken without fully
    loading pixels - cheap and safe for a first pass.
    """
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def load_rgb(data: bytes) -> Image.Image:
    """Decode bytes into an RGB PIL image (CLIP expects 3 channels)."""
    im = Image.open(io.BytesIO(data))
    return im.convert("RGB")


def make_thumbnail(data: bytes, out_dir: Path, image_id: str,
                   size: int = config.THUMB_SIZE) -> Path:
    """Write a JPEG thumbnail (<= size x size) and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    im = load_rgb(data)
    im.thumbnail((size, size))  # in place, preserves aspect ratio
    thumb_path = out_dir / f"{image_id}.jpg"
    im.save(thumb_path, format="JPEG", quality=85)
    return thumb_path


def save_original(data: bytes, out_dir: Path, image_id: str, filename: str) -> Path:
    """Persist the full-size image so the API can serve it later."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(filename).suffix.lower() or ".jpg"
    dest = out_dir / f"{image_id}{ext}"
    dest.write_bytes(data)
    return dest


def prepare(item: ImageItem) -> dict | None:
    """Validate + persist one item; return its payload metadata, or None if bad.

    The returned dict becomes the Qdrant payload (searchable metadata) for this
    image. Returning None signals "skip this file" to the caller.
    """
    if not validate_image(item.data):
        # Save the bytes aside so we can inspect / repair later.
        config.QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        (config.QUARANTINE_DIR / item.filename).write_bytes(item.data)
        return None

    original = save_original(item.data, config.IMAGES_DIR, item.id, item.filename)
    thumb = make_thumbnail(item.data, config.THUMBS_DIR, item.id)
    width, height = load_rgb(item.data).size

    return {
        "filename": item.filename,
        "path": str(original),
        "thumb_path": str(thumb),
        "width": width,
        "height": height,
        "tags": [],
        **item.meta,
    }
