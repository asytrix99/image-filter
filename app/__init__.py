"""image-filter: semantic image search over a collection of images.

Phase 1 modules:
    config   - central settings
    sources  - where images come from (ZipSource / FolderSource)
    ingest   - validate + thumbnail images
    embed    - CLIP model: image/text -> vectors (PyTorch)
    store    - vector store (Qdrant) behind a VectorStore interface
    search   - text/image search on top of the store
"""

__version__ = "0.1.0"
