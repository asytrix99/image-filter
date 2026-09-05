"""Phase 1 CLI: query the index from the terminal.

Usage (from image-filter/):
    ./.venv/Scripts/python.exe -m scripts.query "baby"
    ./.venv/Scripts/python.exe -m scripts.query "dog on a beach" --k 10
    ./.venv/Scripts/python.exe -m scripts.query --image path/to/photo.jpg

Prints the top matches (filename + score) so we can eyeball ranking quality
before any web UI exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app import search


def main() -> None:
    ap = argparse.ArgumentParser(description="Search the image index.")
    ap.add_argument("query", nargs="?", help="text query, e.g. \"baby\"")
    ap.add_argument("--image", help="path to an example image ('find more like this')")
    ap.add_argument("--k", type=int, default=10, help="number of results")
    args = ap.parse_args()

    if args.image:
        results = search.search_by_image(Path(args.image).read_bytes(), top_k=args.k)
        header = f"Top {args.k} similar to image {args.image!r}:"
    elif args.query:
        results = search.search(args.query, top_k=args.k)
        header = f"Top {args.k} matches for {args.query!r}:"
    else:
        raise SystemExit("provide a text query or --image <path>")

    print(header)
    for rank, (image_id, score, payload) in enumerate(results, 1):
        name = payload.get("filename", image_id)
        print(f"  {rank:2d}. {score:.3f}  {name}")


if __name__ == "__main__":
    main()
