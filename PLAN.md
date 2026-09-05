# Plan: Image Filter — semantic image search from a ZIP

## Context

You want a web app where a user uploads a ZIP of ~1000 images, the app "understands"
each image with computer vision, and a search bar (e.g. type `baby`) instantly returns
the matching images. Tech stack: **Strands SDK** (AWS Bedrock backend, following the
memory_agent example + your `my_agent/agent.py`) and **PyTorch** for the vision core.

Decisions locked in with you:
- **CV approach:** CLIP semantic search — encode each image once into a vector, encode
  the text query into the same space, rank by cosine similarity. Open-vocabulary, so any
  word/phrase works (not a fixed label list). This is the "type anything, get matches
  instantly" experience.
- **LLM backend:** AWS Bedrock (`BedrockModel` + boto3), matching your example.
- **Build order:** Phase 1 core pipeline (CLI, verifiable) → Phase 2 web app → Phase 3
  Strands agent layer.

The current `image-filter/` directory is empty (greenfield).

## Key technical decisions (with rationale)

1. **Model: `open_clip` ViT-B/32 (pretrained `laion2b_s34b_b79k`).** Small, fast on CPU,
   great zero-shot quality. We use a *pretrained* model (no training/fine-tuning needed);
   "learning" here = the pretrained CLIP embeddings. Alternative `transformers` CLIP also
   works; `open_clip` is cleaner for pure embedding.
2. **Vector store: Qdrant (scales to millions), behind a `VectorStore` interface.**
   You want this to handle *a lot* of images and later connect to the cloud, so we skip the
   NumPy/`.npy` brute-force and use a real ANN vector DB from the start:
   - **Local dev:** Qdrant **embedded** mode — `QdrantClient(path="./data/qdrant")` — on-disk,
     no server, no Docker, Windows-friendly (`:memory:` for tests).
   - **Scale / cloud:** the *same* `qdrant-client` API points at a Qdrant server (Docker) or
     **Qdrant Cloud** by changing only a URL — HNSW ANN index, millions of vectors, filtered
     search. This directly matches the "connect to cloud later" goal.
   - Qdrant stores the embedding **plus a metadata payload** (filename, tags, dims, source),
     so it is a **single store** for vectors *and* metadata and supports tag-filtered search.
   - All access goes through a small `VectorStore` abstraction (`upsert`, `search`, `get`,
     `count`) so it can be swapped for **pgvector** (if you later want one relational DB for
     everything) or FAISS without touching the rest of the app.
3. **Ingestion source is pluggable (`Source` interface) — ZIP now, MCP/cloud later.**
   Module 1 reads images through a `Source` that yields `(id, bytes, metadata)`. v1 ships
   `ZipSource`; a future `McpSource` lists/fetches images from a cloud store (S3, Drive, …)
   via an MCP server — verified supported by `strands.tools.mcp.MCPClient`. No zip needed in
   that iteration; the rest of the pipeline (embed → store → search) is unchanged.
4. **⚠️ Python version.** The shared `.venv` is **Python 3.14.6**; PyTorch/open_clip
   generally lag new Python releases and may lack 3.14 wheels. **Recommendation: create a
   dedicated project venv on Python 3.11 or 3.12** for `image-filter` so torch installs
   cleanly. First implementation step is to confirm `pip install torch open_clip_torch`
   succeeds; if 3.14 fails, fall back to 3.12.

## Directory structure

```
image-filter/
  pyproject.toml / requirements.txt
  README.md
  app/
    __init__.py
    config.py            # paths, model name, batch size, Qdrant url/path, Bedrock model id/region
    sources.py           # Module 1: Source interface + ZipSource (McpSource later)
    ingest.py            # Module 1: validate, dedupe, thumbnails (reads via a Source)
    embed.py             # Module 2: CLIP model load + batch image/text encoding (PyTorch)
    store.py             # Module 3: VectorStore interface + QdrantStore (base+adapted vectors)
    search.py            # Module 4: text/image -> vector, ANN search, filters, zero-shot tags
    feedback.py          # Module 8: feedback store + precision@k metric + threshold logic
    train.py             # Module 8: tiered training (rerank / adapter / LoRA), bg job, swap
    agents/              # Module 6: multi-agent system (Strands + Bedrock)
      __init__.py
      shared.py          #   Bedrock model factory + shared @tool wrappers
      orchestrator.py    #   top-level router (agents-as-tools)
      ingestion_agent.py #   extraction, retry-on-corrupt, batching strategy (+ Graph)
      search_agent.py    #   query interpretation, vector search, re-rank
      curator_agent.py   #   auto-tag, cluster naming, dedupe, collection stats
      training_agent.py  #   feedback loop, accuracy check, human-in-the-loop retrain trigger
    api.py               # Module 5: FastAPI app (upload, search, image, thumb, ask, feedback)
    web/
      index.html         # Module 7: upload widget + search bar + results grid
  scripts/
    build_index.py       # CLI: point at a zip/folder -> builds the index (Phase 1 demo)
    query.py             # CLI: query the index from terminal (Phase 1 demo)
  data/                  # gitignored: images, thumbnails, qdrant/, adapters/ (versioned), feedback.sqlite
  tests/
    test_ingest.py
    test_search.py
    test_feedback.py
```

## Modules

### Module 1 — Ingestion (`app/sources.py` + `app/ingest.py`)
- **`Source` interface (`sources.py`)** — `iter_images() -> Iterator[(id, bytes, meta)]`
  and `count()`. Decouples *where images come from* from the pipeline:
  - `ZipSource(zip_path)` (v1): safe unzip (guard zip-slip / path traversal, cap total
    size + file count), yield only image entries.
  - `FolderSource(dir)` (trivial, handy for the CLI demo).
  - `McpSource(...)` (future): lists/fetches from a cloud store via `strands.tools.mcp.
    MCPClient` — no zip needed. Same downstream pipeline.
- `validate_image(bytes) -> bool`: open with Pillow, verify it decodes; skip/quarantine
  corrupt files.
- `make_thumbnail(bytes, thumb_dir, size=256) -> Path`: Pillow thumbnail for fast grid.
- Optional perceptual-hash dedupe.
- Produces per-image `(payload metadata, image bytes)` handed to embed + store.

### Module 2 — Embedding / CV core (`app/embed.py`, PyTorch)
- `load_model()`: load `open_clip` ViT-B/32 + preprocess transform once (cache globally).
- `embed_images(paths, batch_size=32) -> np.ndarray`: batched, `torch.no_grad()`,
  normalized float32 vectors; auto-select CUDA if available else CPU.
- `embed_text(query) -> np.ndarray`: tokenize + encode + normalize.
- This is the "PyTorch computer vision" component.
- **Two-stage embedding to support cheap adaptation (see Module 8):** keep the CLIP backbone
  **frozen** and cache its *base* embeddings; apply a small trainable **adapter** on top
  (`apply_adapter(base_vec) -> adapted_vec`). Search always runs on adapted vectors. Because
  the backbone is frozen, "retraining" only updates the tiny adapter — re-applying it to the
  cached base embeddings is a fast matmul, NOT a full re-encode of every image. `config` has
  an `ADAPTER_VERSION`; the active adapter is loaded here.

### Module 3 — Storage (`app/store.py`, Qdrant behind a `VectorStore` interface)
- **`VectorStore` interface:** `upsert(ids, vectors, payloads)`, `search(vector, k, filter)`,
  `get(id)`, `count()` — so the backend is swappable (Qdrant → pgvector/FAISS) without
  touching callers.
- **`QdrantStore`** (default impl): one collection with `size=512, distance=Cosine`.
  Each point = `id` + embedding + **payload** `{filename, thumb_path, width, height, tags,
  source}`. Vectors AND metadata live together, so no separate SQLite is needed and
  tag-filtered search comes free.
  - Local: `QdrantClient(path=config.QDRANT_PATH)` (embedded, on-disk, persists across
    restarts). Cloud: `QdrantClient(url=..., api_key=...)` — same code.
- **Named vectors for adaptation (Module 8):** store two vectors per point — `base` (frozen
  CLIP output, not searched) and `adapted` (searched). After the adapter retrains, a fast
  background pass reads all `base` vectors, applies the new adapter, and upserts new `adapted`
  vectors — no image re-encoding. Blue/green: write into a new collection and swap atomically
  so live search never breaks.
- Batched `upsert` during ingest; `search` delegates to Qdrant's HNSW ANN index (on `adapted`).

### Module 4 — Search (`app/search.py`)
- `search(query, top_k=50, filter=None) -> list[(image_id, score, payload)]`: `embed_text`
  -> `VectorStore.search(vec, k, filter)` (Qdrant ANN cosine). Optional `filter` maps to a
  Qdrant payload filter (e.g. only certain tags).
- **`search_by_image(image_bytes, top_k=50) -> list[(image_id, score, payload)]`** — the
  "find more like this" core. Reuses `embed_images` (Module 2) on the single query image,
  then runs the *identical* `VectorStore.search`. Works because CLIP puts images and text in
  the same space, so text-search and image-search share one code path. Optionally exclude the
  query image itself if it's already in the index.
- `search_combined(image_bytes, text, top_k)` (nice-to-have) — average/weight the image and
  text vectors ("more like this, but outdoors") before searching.
- Optional `auto_tag(image_vec, label_set) -> list[str]`: zero-shot classify against a
  candidate label list for a browsable tag chip UI (nice-to-have).

### Module 5 — Web API (`app/api.py`, FastAPI + uvicorn — uvicorn already installed)
- `POST /upload` — accept ZIP, run ingest+embed+store as a background task, return a job id.
- `GET /status/{job_id}` — indexing progress (for the 1000-image case).
- `GET /search?q=&k=` — text search: ranked image ids + scores + thumb URLs.
- `POST /search_by_image` — multipart image upload ("find more like this"): calls
  `search_by_image` (Module 4). Optional `q` field to combine image + text.
- `GET /thumb/{id}` and `GET /image/{id}` — serve files.
- `POST /ask` — natural-language endpoint routed through the Strands agent (Module 6).
- `POST /feedback` — record relevance labels + notes (Module 8).
- `GET /metrics` — current precision@k / whether retraining is suggested.
- `POST /train` — confirm & kick off background retraining; `GET /train/status` — job state
  (`idle|training|swapping|ready`) for the UI status chip (Module 8).
- Serves `web/index.html` at `/`.

### Module 6 — Multi-agent system (`app/agents/`, Strands + Bedrock)

Goal: maximize Strands capabilities with a **team of specialized agents** coordinated by an
orchestrator, instead of one flat agent. Backend: `BedrockModel(model_id=..., region_name=...)`
from config (mirrors `my_agent/agent.py`), shared by all agents.

**Verified against the installed SDK (`strands==1.54.0`):** `strands.multiagent` exports
`GraphBuilder` (DAG), `Swarm` (autonomous handoff), and `Agent.as_tool()` (agents-as-tools).
We use **agents-as-tools** for top-level routing and a **Graph** for the deterministic ingest
pipeline.

**Design principle (important — cost/latency):** the LLM agents make *decisions*
(batch size, whether/how to retry, query rewriting, re-ranking), but the *heavy per-image
loops* (decode, embed 1000 images) stay in plain Python tools from Modules 1–4. Agents must
never loop once-per-image through the model — that would be slow and expensive. Each agent is
`Agent(model=bedrock_model, tools=[...], system_prompt=...)` wrapping existing pipeline
functions as `@tool`s.

**The agents (`app/agents/`):**

1. **Orchestrator (`orchestrator.py`)** — top-level entry for `/ask`. Uses agents-as-tools:
   `Agent(system_prompt="route to specialists...", tools=[ingestion_agent.as_tool(...),
   search_agent.as_tool(...), curator_agent.as_tool(...)])`. Routes a user request to the
   right specialist and composes the final answer. Use `.as_tool(delegate=True)` for the
   search agent so image results pass straight through without an extra round-trip.

2. **Ingestion agent (`ingestion_agent.py`)** — tools wrapping Module 1/2/3:
   `extract_zip`, `validate_image`, `repair_or_retry(path)` (re-decode with fallbacks, e.g.
   truncated-image tolerance, format re-encode), `plan_batches(count, device, mem)` (decides
   batch size / whether to use GPU), `embed_batch`, `persist`. The agent reasons about
   failures: retries corrupted files with fallbacks, quarantines the unrecoverable, and picks
   a batching strategy — exactly your spec. Internally the fixed happy-path sequence
   (extract → validate → batch → embed → persist) is modeled as a **Strands `Graph`**
   (`GraphBuilder`) for determinism; the agent supervises and handles the exceptional paths.

3. **Search agent (`search_agent.py`) — multimodal (text AND image input)** — tools:
   `interpret_query(q)` (rewrite/expand, e.g. "cute lil babies" → {"baby","infant","toddler"},
   extract filters like "outdoor"), `vector_search(query, k)` (Module 4),
   **`search_by_image(image_bytes, k)`** (Module 4, "find more like this"),
   `rerank(candidates, intent)`, `format_results`. Accepts a Strands multimodal message: a
   `ContentBlock` list mixing text and an `image` block
   (`{"image": {"format": "jpeg", "source": {"bytes": ...}}}`) — verified present in
   `strands.types.media.ImageContent`. Three query modes through the same agent:
   - **text only** → `interpret_query` → `vector_search` → `rerank`.
   - **image only** ("find more like this", drag a photo) → `search_by_image` directly.
   - **image + text** ("more like this but outdoors") → agent (multimodal Bedrock model,
     e.g. Claude Sonnet) reads the image + text, extracts intent, calls `search_by_image`
     and/or `search_combined`, then `rerank`.
   *Guardrail:* for pure image similarity the LLM does **not** need to see the pixels — the
   CLIP `search_by_image` tool embeds and searches directly; only route the image *through*
   the model when text is combined with it and intent must be reasoned about. This keeps the
   cheap path cheap.

4. **Curator agent (`curator_agent.py`)** — the "other agents": zero-shot `auto_tag`,
   `name_clusters` (label groups of similar images), `find_duplicates`, and
   `describe_collection` (counts / sample tags for "how many dogs?"-style questions).

5. **Feedback/Training agent (`training_agent.py`)** — owns the Module 8 loop: tools
   `record_feedback`, `evaluate_accuracy` (precision@k vs threshold), `recommend_training`,
   and `start_training(tier)`. Uses a Strands **human-in-the-loop interrupt** to ask the user
   to confirm retraining before spending compute, then launches/monitors the background job.
   It decides the tier (re-rank → adapter → LoRA) based on sample count and metric.

**Shared plumbing (`app/agents/shared.py`):** one `build_bedrock_model()` factory + the
common `@tool` wrappers, so agents reuse the same pipeline functions (no duplication).

### Module 7 — Frontend (`app/web/index.html`)
- Minimal single-page: drag-drop ZIP upload with progress, a search bar, a responsive
  results grid of thumbnails, click-to-enlarge. Plain HTML+JS (no build step); fetches the
  FastAPI endpoints. Can upgrade to React later if desired.
- **Search-by-example:** drop a photo onto the search bar (or a "find more like this" button
  on any grid thumbnail) → POST to `/search_by_image` → same results grid. One input area
  accepts either typed text or a dropped image (or both).
- **Feedback tab (new):** after results render, an "Is this accurate? Let us know…" panel —
  per-result 👍/👎 (relevant/not) plus a free-text box ("what were you looking for?"). Posts
  to `/feedback`. A persistent status chip shows the model state: `ready` / `retraining…`.
  While retraining, search still works (uses the current adapter); when it flips back to
  `ready`, a "model updated — try again" toast invites the user to re-search, and if still
  wrong they give more feedback → loop.

### Module 8 — Feedback & adaptive retraining (`app/feedback.py` + `app/train.py`)

Human-in-the-loop loop: collect relevance feedback → measure accuracy → when it drops below a
threshold *and* enough new samples exist, ask the user to confirm retraining → train in the
background (search stays live) → swap in the improved model → user retries.

- **Feedback store (`feedback.py`)** — a small table (Qdrant payload or a `feedback.sqlite`):
  `(query_text or query_image_ref, image_id, label ∈ {relevant, not_relevant}, note, ts)`.
- **Accuracy metric** — rolling **precision@k** over recent feedback (fraction of shown
  results marked relevant) and/or a satisfaction score. Exposed via `/metrics`.
- **Threshold + human-in-the-loop trigger** — when `precision@k < THRESHOLD` **and**
  `new_labeled_samples >= MIN_SAMPLES`, the app surfaces "Accuracy looks low — retrain now?"
  (decided *with* the user, per your spec). Confirmation can flow through a Strands
  human-in-the-loop **interrupt** in the Feedback/Training agent, or a plain UI confirm.
- **Tiered training (`train.py`) — escalate only as needed, cheapest first:**
  1. **Re-rank only (no training):** immediately use feedback to boost/penalize results.
     Zero cost, no re-embedding. Always on.
  2. **Adapter fine-tune (default "retraining"):** train the small adapter MLP on the frozen
     CLIP embeddings with a contrastive/triplet loss (relevant image ↔ query pulled together,
     not-relevant pushed apart). PyTorch training, but tiny and fast; backbone frozen ⇒ only
     `base`→`adapted` vectors get recomputed (fast matmul, Module 3 blue/green swap).
  3. **LoRA backbone fine-tune (gated):** only when adapter isn't enough *and* enough data
     accumulated. LoRA/PEFT on the CLIP image encoder → requires a **full background
     re-embed** of the corpus into a new collection (expensive) → blue/green swap.
- **Background execution** — training runs as a background job (FastAPI `BackgroundTasks` /
  asyncio worker). `/train/status` returns `idle|training|swapping|ready`. Search continues
  against the current adapter/collection throughout; the new one is swapped in atomically on
  completion.
- **Validation & rollback (critical guardrail)** — hold out a slice of feedback as a
  validation set; only promote a newly trained adapter/model if precision@k **improves** on
  it, else discard and keep the previous version. Adapter/model versions are kept so we can
  roll back. This prevents a few noisy clicks from making search worse.
- **Cold-start note:** with very few labels, Tiers 2–3 can't meaningfully help; the system
  stays on Tier 1 re-ranking until `MIN_SAMPLES` is reached.

## Build phases

- **Phase 1 (core, verifiable in terminal):** Modules 1–4 + `scripts/build_index.py` and
  `scripts/query.py`. Prove: unzip 1000 imgs → build index → `python scripts/query.py baby`
  prints the top matching filenames. No web, no agent yet.
- **Phase 2 (web):** Module 5 API + Module 7 UI. Upload a zip in the browser, search, see
  the grid.
- **Phase 3 (multi-agent):** Module 6. Build specialists incrementally — search_agent first
  (wire `/ask` to it), then curator_agent, then ingestion_agent (route `/upload` through it),
  then the orchestrator tying them together with agents-as-tools. Chat box in the UI hits
  `/ask`.
- **Phase 4 (feedback + adaptive retraining):** Module 8, layered by tier. First the feedback
  tab + store + precision@k metric + Tier-1 re-ranking (no training). Then the frozen-backbone
  **adapter** training as a background job with the `retraining…` status and blue/green vector
  swap, plus validation/rollback. LoRA backbone fine-tune (Tier 3) is optional/last. Add the
  training_agent human-in-the-loop trigger once the mechanics work.

## Verification

- **Env sanity:** in a fresh 3.11/3.12 venv, `pip install -r requirements.txt` (torch,
  open_clip_torch, qdrant-client, fastapi, strands-agents, pillow; + `peft`/`accelerate` only
  if enabling Tier-3 LoRA); confirm
  `python -c "import torch, open_clip, qdrant_client; print(torch.__version__)"`.
- **Phase 1:** create a tiny test zip of ~10 varied images; run `build_index.py`; run
  `query.py "dog"`, `query.py "beach"` and confirm relevant files rank first, then re-run to
  confirm the embedded Qdrant persisted (no re-index). Add `tests/test_ingest.py` (zip-slip
  rejected, corrupt image skipped) and `tests/test_search.py` (known image ranks top for its
  obvious query; uses Qdrant `:memory:`).
- **Phase 2:** `uvicorn app.api:app --reload`, open `/`, upload the test zip, watch
  `/status`, search `baby`, confirm thumbnails render. Then drop one of the indexed images
  onto the search bar and confirm `/search_by_image` returns visually similar images
  ("find more like this") — the query image should rank itself/near-duplicates first.
- **Phase 3:** with AWS creds configured — (a) search_agent: POST `/ask` "show me babies
  outdoors" and confirm interpret→search→rerank runs and thumbnails come back;
  (b) curator/orchestrator: "how many dogs?" routes to curator's `describe_collection`;
  (c) ingestion_agent: upload a zip containing a deliberately corrupted file and confirm the
  agent retries, quarantines the unrecoverable one, and still indexes the rest. Log each
  agent's tool calls to verify routing.
- **Phase 4:** search a query, mark results 👎 in the feedback tab; confirm `/metrics`
  precision@k drops and a retrain suggestion appears. Trigger `/train`; confirm the status
  chip shows `retraining…` **while search still returns results**, then flips to `ready`.
  Re-run the same query and confirm the previously-👎 results are down-ranked / better ones
  surface. Verify the validation guard: feed contradictory/noisy labels and confirm a
  non-improving adapter is **rejected** (rolls back to the prior version). `test_feedback.py`
  covers metric math + threshold trigger + rollback-on-no-improvement.

## Risks / open items
- Python 3.14 torch wheels (see decision #4) — resolve at first install step.
- AWS Bedrock model id + region must be set (reuse the ids from `my_agent/agent.py`);
  needs valid credentials for Phase 3. The search agent's image+text mode needs a
  **multimodal** Bedrock model (e.g. Claude Sonnet) — but pure image-similarity and pure
  text search don't route through the model at all, so they work regardless.
- Large uploads: cap zip size and image count in `ingest.py`; indexing many images on CPU
  can be slow — that's why upload is a background job with `/status`. At large scale, move
  Qdrant from embedded to a server/cloud instance and consider a GPU for embedding.
- **Adaptive retraining caveats (Module 8):** (1) fine-tuning the CLIP *backbone* (Tier 3)
  invalidates all stored embeddings and forces a full re-embed — the frozen-backbone adapter
  (Tier 2) avoids this and is the default. (2) A handful of clicks can't safely retrain a
  large model; the `MIN_SAMPLES` gate + validation/rollback exist to stop feedback from making
  search *worse*. (3) On CPU, adapter training is quick; Tier-3 LoRA realistically wants a GPU.
  This is the most experimental module — Phase 4, after core search is solid.
- **Future — cloud source via MCP:** the `Source` interface (Module 1) + `strands.tools.mcp.
  MCPClient` (verified installed) mean a later iteration can add `McpSource` to pull images
  directly from a cloud store through an MCP server — no zip. The embed → Qdrant → search
  path stays identical, and Qdrant can already be pointed at Qdrant Cloud. Kept out of v1
  scope but the design leaves the seams for it.
