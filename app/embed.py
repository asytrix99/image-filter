"""Module 2 - the CLIP embedding core (PyTorch).

CLIP maps images AND text into the same 512-dim vector space, so a text query
and a matching image land near each other. That single property is what powers
both "search by word" and "find more like this".

The model is loaded once and cached. All heavy work happens here in plain
PyTorch - no LLM involved.
"""

from __future__ import annotations

import io
from functools import lru_cache
from typing import Iterable

import numpy as np
import open_clip
import torch
from PIL import Image

from . import config


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def load_model():
    """Load (and cache) the CLIP model, preprocess transform, and tokenizer.

    lru_cache makes this a lazy singleton: the first call loads weights (may
    download on first ever run), later calls return the same objects instantly.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED
    )
    model = model.to(_device()).eval()
    tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)
    return model, preprocess, tokenizer


def _normalize(v: torch.Tensor) -> torch.Tensor:
    """L2-normalize so a dot product equals cosine similarity."""
    return v / v.norm(dim=-1, keepdim=True)


def embed_images(images: Iterable[bytes | Image.Image],
                 batch_size: int = config.EMBED_BATCH_SIZE) -> np.ndarray:
    """Embed a batch of images -> (N, 512) float32 array of unit vectors.

    Accepts raw bytes or PIL images. Runs under torch.no_grad() (inference only)
    and in batches to keep memory bounded for large collections.
    """
    model, preprocess, _ = load_model()
    device = _device()

    tensors: list[torch.Tensor] = []
    for img in images:
        if isinstance(img, (bytes, bytearray)):
            img = Image.open(io.BytesIO(img)).convert("RGB")
        tensors.append(preprocess(img))

    if not tensors:
        return np.zeros((0, config.EMBED_DIM), dtype=np.float32)

    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i:i + batch_size]).to(device)
            feats = model.encode_image(batch)
            feats = _normalize(feats)
            out.append(feats.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def embed_text(text: str | list[str]) -> np.ndarray:
    """Embed one or more text queries -> (N, 512) float32 unit vectors."""
    model, _, tokenizer = load_model()
    device = _device()
    texts = [text] if isinstance(text, str) else text

    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = _normalize(feats)
    return feats.cpu().numpy().astype(np.float32)
