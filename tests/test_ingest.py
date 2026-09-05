"""Tests for Module 1 (sources + ingest): safety and validation."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from PIL import Image

from app import ingest
from app.sources import ZipSource


def _png_bytes(color=(255, 0, 0), size=(32, 32)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_validate_accepts_real_image_rejects_garbage():
    assert ingest.validate_image(_png_bytes()) is True
    assert ingest.validate_image(b"this is not an image") is False


def test_zipsource_skips_zip_slip_and_nonimages(tmp_path: Path):
    zip_path = tmp_path / "in.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("good.png", _png_bytes())          # kept
        zf.writestr("../evil.png", _png_bytes())       # zip-slip -> rejected
        zf.writestr("notes.txt", b"hello")             # non-image -> skipped
    src = ZipSource(zip_path)
    names = [item.filename for item in src.iter_images()]
    assert names == ["good.png"]
    assert src.count() == 1


def test_content_addressed_id_is_stable(tmp_path: Path):
    data = _png_bytes()
    zip_path = tmp_path / "dup.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("a.png", data)
        zf.writestr("b.png", data)  # same bytes, different name
    ids = [item.id for item in ZipSource(zip_path).iter_images()]
    assert ids[0] == ids[1]  # identical content -> identical id (free dedup)
