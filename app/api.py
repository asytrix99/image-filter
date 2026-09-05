"""Module 5 - FastAPI web backend.

Endpoints:
    GET  /                  -> the web UI (web/index.html)
    POST /upload            -> accept a .zip, index it in the background, return job_id
    GET  /status/{job_id}   -> indexing progress
    GET  /search?q=&k=      -> text search
    POST /search_by_image   -> "find more like this" (multipart image, optional q text)
    GET  /thumb/{image_id}  -> thumbnail JPEG
    GET  /image/{image_id}  -> full-size image
    GET  /stats             -> collection size

Run (from image-filter/):
    ./.venv/Scripts/python.exe -m uvicorn app.api:app --reload

Note: embedded Qdrant locks its storage dir, so we create ONE shared QdrantStore
and pass it into every search/index call (never construct a second client on the
same path).
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from . import config, indexer, search
from .sources import ZipSource
from .store import get_store

app = FastAPI(title="image-filter")

config.ensure_dirs()
STORE = get_store()                   # single shared client (see module docstring)
JOBS: dict[str, dict] = {}            # in-memory job status: job_id -> {...}
WEB_DIR = Path(__file__).parent / "web"


# --- UI ---------------------------------------------------------------------
@app.get("/")
def home():
    return FileResponse(WEB_DIR / "index.html")


# --- Upload + background indexing -------------------------------------------
def _run_index_job(job_id: str, zip_path: str) -> None:
    try:
        JOBS[job_id]["status"] = "indexing"

        def prog(indexed: int, skipped: int, total: int) -> None:
            JOBS[job_id].update(indexed=indexed, skipped=skipped, total=total)

        result = indexer.index_source(ZipSource(zip_path), store=STORE, on_progress=prog)
        JOBS[job_id].update(status="done", **result)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the client
        JOBS[job_id].update(status="error", error=str(exc))
    finally:
        try:
            Path(zip_path).unlink()
        except OSError:
            pass


@app.post("/upload")
async def upload(background: BackgroundTasks, file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(400, "please upload a .zip file")

    tmp = Path(tempfile.gettempdir()) / f"imgf_{uuid.uuid4().hex}.zip"
    with tmp.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "queued", "indexed": 0, "skipped": 0, "total": 0}
    background.add_task(_run_index_job, job_id, str(tmp))
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "unknown job")
    return JOBS[job_id]


# --- Search -----------------------------------------------------------------
def _to_results(hits) -> list[dict]:
    return [
        {
            "image_id": image_id,
            "score": round(float(score), 4),
            "filename": (payload or {}).get("filename"),
            "thumb_url": f"/thumb/{image_id}",
            "image_url": f"/image/{image_id}",
        }
        for image_id, score, payload in hits
    ]


@app.get("/search")
def search_text(q: str, k: int = 50):
    hits = search.search(q, top_k=k, store=STORE)
    return {"query": q, "results": _to_results(hits)}


@app.post("/search_by_image")
async def search_image(file: UploadFile = File(...),
                       q: str | None = Form(None), k: int = Form(50)):
    data = await file.read()
    if q:
        hits = search.search_combined(data, q, top_k=k, store=STORE)
    else:
        hits = search.search_by_image(data, top_k=k, store=STORE)
    return {"results": _to_results(hits)}


# --- File serving -----------------------------------------------------------
@app.get("/thumb/{image_id}")
def thumb(image_id: str):
    path = config.THUMBS_DIR / f"{image_id}.jpg"
    if not path.is_file():
        raise HTTPException(404, "thumbnail not found")
    return FileResponse(path)


@app.get("/image/{image_id}")
def image(image_id: str):
    matches = list(config.IMAGES_DIR.glob(f"{image_id}.*"))
    if not matches:
        raise HTTPException(404, "image not found")
    return FileResponse(matches[0])


@app.get("/stats")
def stats():
    return {"count": STORE.count()}


# --- Natural-language assistant (Module 6, needs AWS Bedrock creds) ----------
_ORCHESTRATOR = None


def _get_orchestrator():
    """Build the orchestrator on first use (so the server starts without AWS)."""
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        from .agents.orchestrator import build_orchestrator
        _ORCHESTRATOR = build_orchestrator()
    return _ORCHESTRATOR


@app.post("/ask")
def ask(q: str = Body(..., embed=True)):
    """Route a natural-language request through the Strands multi-agent system.

    Returns the assistant's text answer plus any images its tools surfaced.
    """
    q = (q or "").strip()
    if not q:
        raise HTTPException(400, "empty question")

    from .agents import shared
    shared.reset_results()
    try:
        result = _get_orchestrator()(q)
    except Exception as exc:  # noqa: BLE001 - surface agent/Bedrock errors to the UI
        raise HTTPException(502, f"agent error (is AWS Bedrock configured?): {exc}")

    return {"answer": str(result), "results": shared.take_results()}
