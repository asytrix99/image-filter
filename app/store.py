"""Module 3 - the vector store (Qdrant behind a small interface).

`VectorStore` is the only thing the rest of the app talks to, so the backend
(Qdrant now; pgvector/FAISS later) can be swapped without touching callers.

Qdrant holds, per image: the id, the embedding, and a payload (metadata like
filename / thumb_path / tags). Vectors + metadata live together, so there is no
separate SQL database to keep in sync, and tag-filtered search comes for free.

Local dev uses embedded, on-disk Qdrant (config.QDRANT_PATH). Setting
IMGF_QDRANT_URL switches to a Qdrant server / Qdrant Cloud with identical code.
"""

from __future__ import annotations

import uuid
from typing import Protocol

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from . import config


class VectorStore(Protocol):
    def upsert(self, ids: list[str], vectors: np.ndarray,
               payloads: list[dict]) -> None: ...
    def search(self, vector: np.ndarray, top_k: int = 50,
               query_filter=None) -> list[tuple[str, float, dict]]: ...
    def get(self, image_id: str) -> dict | None: ...
    def count(self) -> int: ...


class QdrantStore:
    """Default VectorStore backed by Qdrant."""

    def __init__(self, collection: str = config.QDRANT_COLLECTION,
                 client: QdrantClient | None = None):
        self.collection = collection
        if client is not None:
            # Injected client (e.g. QdrantClient(location=":memory:") in tests).
            self.client = client
        elif config.QDRANT_URL:
            # Cloud / server if a URL is configured.
            self.client = QdrantClient(
                url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY
            )
        else:
            # Local embedded, on-disk storage (default for dev).
            config.ensure_dirs()
            self.client = QdrantClient(path=config.QDRANT_PATH)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=config.EMBED_DIM, distance=Distance.COSINE
                ),
            )

    @staticmethod
    def _point_id(image_id: str) -> str:
        # Qdrant point ids must be an int or a UUID string. Deterministically
        # map any image id (sha1 hex, or an arbitrary test string) to a stable
        # UUID via uuid5, so the same image id always yields the same point id.
        return str(uuid.uuid5(uuid.NAMESPACE_URL, image_id))

    def upsert(self, ids: list[str], vectors: np.ndarray,
               payloads: list[dict]) -> None:
        points = [
            PointStruct(
                id=self._point_id(iid),
                vector=vec.tolist(),
                payload={**pl, "image_id": iid},
            )
            for iid, vec, pl in zip(ids, vectors, payloads)
        ]
        if points:
            self.client.upsert(collection_name=self.collection, points=points)

    def search(self, vector: np.ndarray, top_k: int = 50,
               query_filter=None) -> list[tuple[str, float, dict]]:
        res = self.client.query_points(
            collection_name=self.collection,
            query=vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points
        return [(p.payload.get("image_id", str(p.id)), p.score, p.payload) for p in res]

    def get(self, image_id: str) -> dict | None:
        res = self.client.retrieve(
            collection_name=self.collection,
            ids=[self._point_id(image_id)],
            with_payload=True,
        )
        return res[0].payload if res else None

    def count(self) -> int:
        return self.client.count(collection_name=self.collection).count
