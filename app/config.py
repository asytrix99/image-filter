"""Central configuration for image-filter.

Everything else imports from here so paths, model choice, and the vector-store
location live in exactly one place. Values can be overridden with environment
variables (handy for pointing at Qdrant Cloud later without touching code).
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Filesystem layout -------------------------------------------------------
# Project root = the image-filter/ directory (this file is app/config.py).
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("IMGF_DATA_DIR", ROOT_DIR / "data"))

IMAGES_DIR = DATA_DIR / "images"        # extracted full-size images
THUMBS_DIR = DATA_DIR / "thumbs"        # generated thumbnails (fast grid)
QUARANTINE_DIR = DATA_DIR / "quarantine"  # files we could not decode
QDRANT_PATH = str(DATA_DIR / "qdrant")  # embedded, on-disk Qdrant storage

# --- CLIP model (Module 2) ---------------------------------------------------
# ViT-B/32 is small, fast on CPU, and outputs 512-dim embeddings.
CLIP_MODEL_NAME = os.getenv("IMGF_CLIP_MODEL", "ViT-B-32")
CLIP_PRETRAINED = os.getenv("IMGF_CLIP_PRETRAINED", "laion2b_s34b_b79k")
EMBED_DIM = 512                          # must match the chosen model's output
EMBED_BATCH_SIZE = int(os.getenv("IMGF_BATCH_SIZE", "32"))

# --- Vector store (Module 3) -------------------------------------------------
# Local embedded mode by default. Set IMGF_QDRANT_URL to use a server / Qdrant
# Cloud instead (the client code is identical either way).
QDRANT_URL = os.getenv("IMGF_QDRANT_URL")          # e.g. https://xyz.qdrant.io
QDRANT_API_KEY = os.getenv("IMGF_QDRANT_API_KEY")  # for Qdrant Cloud
QDRANT_COLLECTION = os.getenv("IMGF_COLLECTION", "images")

# --- Ingestion limits (Module 1) --------------------------------------------
# Guard rails so a malicious / huge zip can't exhaust disk or memory.
MAX_IMAGES = int(os.getenv("IMGF_MAX_IMAGES", "20000"))
MAX_ZIP_BYTES = int(os.getenv("IMGF_MAX_ZIP_BYTES", str(5 * 1024**3)))  # 5 GiB
THUMB_SIZE = int(os.getenv("IMGF_THUMB_SIZE", "256"))
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}

# --- Bedrock / Strands (Phase 3, unused in Phase 1) -------------------------
BEDROCK_MODEL_ID = os.getenv(
    "IMGF_BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-6"
)
BEDROCK_REGION = os.getenv("IMGF_BEDROCK_REGION", "ap-southeast-2")


def ensure_dirs() -> None:
    """Create the data directories if they do not exist yet."""
    for d in (DATA_DIR, IMAGES_DIR, THUMBS_DIR, QUARANTINE_DIR):
        d.mkdir(parents=True, exist_ok=True)
