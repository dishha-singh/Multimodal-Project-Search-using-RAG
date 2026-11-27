"""
backend.py - FastAPI backend for the Multi-Modal Product Search project.

Run it from the project folder:
    $env:GEMINI_API_KEY="your-key"        (optional; turns on the AI summary)
    uvicorn backend:app --port 8000

Then open http://127.0.0.1:8000/docs for the interactive API page.

Architecture (one class per responsibility):
    EmbeddingService  - CLIP: text / image  ->  normalised 512-d vector, query fusion
    VectorStore       - Qdrant: nearest-neighbour search with an optional colour filter
    SummaryService    - Gemini: grounded summary of the retrieved products (RAG)
    SearchLogger      - SQLite: one row per search, plus usage statistics
    FastAPI app       - validates requests, orchestrates the services, returns JSON
"""
import io
import os
import sqlite3
import threading
import time
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

HOME = os.path.expanduser("~")
VECTOR_DB_PATH = os.path.join(HOME, "qdrant_store")      # built by build_index.py
LOG_DB_PATH = os.path.join(HOME, "search_log.db")        # created automatically
IMAGE_DIR = "data/images"                                # product photos
COLLECTION = "products"
TOP_K = 5
MAX_UPLOAD_BYTES = 5 * 1024 * 1024                       # 5 MB
MAX_TEXT_LENGTH = 200


# =============================================================== response schemas
class ProductOut(BaseModel):
    id: int
    title: Optional[str] = None
    description: Optional[str] = None
    colour: Optional[str] = None
    article_type: Optional[str] = None
    gender: Optional[str] = None
    score: Optional[float] = None
    image_url: str


class SearchResponse(BaseModel):
    mode: str                          # "text", "image" or "hybrid"
    colour_filter: Optional[str] = None
    retrieval_ms: float                # embedding + vector search
    latency_ms: float                  # whole request, including the summary
    summary: Optional[str] = None
    results: List[ProductOut]


# =============================================================== services
class EmbeddingService:
    """Turns text and images into comparable CLIP vectors."""

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def _normalise(v):
        return v / np.linalg.norm(v)

    def encode_text(self, text: str):
        return self._normalise(self.model.encode(text))

    def encode_image(self, image: Image.Image):
        return self._normalise(self.model.encode(image))

    def fuse(self, vectors):
        """Combine a text vector and an image vector into one query vector."""
        return self._normalise(sum(vectors))


class VectorStore:
    """Qdrant collection of product vectors plus their metadata (payload)."""

    def __init__(self, path: str, collection: str = COLLECTION):
        self.collection = collection
        self.client = QdrantClient(path=path)
        self.colours = self._load_colours()

    def _load_colours(self):
        colours, offset = set(), None
        while True:
            points, offset = self.client.scroll(self.collection, limit=500, offset=offset,
                                                with_payload=True, with_vectors=False)
            for p in points:
                if p.payload.get("colour"):
                    colours.add(p.payload["colour"])
            if offset is None:
                break
        return sorted(colours)

    def search(self, vector, colour: Optional[str] = None, k: int = TOP_K):
        flt = None
        if colour:
            flt = Filter(must=[FieldCondition(key="colour", match=MatchValue(value=colour))])
        return self.client.query_points(self.collection, query=vector.tolist(),
                                        query_filter=flt, limit=k).points

    def get(self, product_id: int):
        found = self.client.retrieve(self.collection, ids=[product_id], with_payload=True)
        return found[0] if found else None

    def count(self) -> int:
        return self.client.count(self.collection).count

    def close(self):
        self.client.close()


class SummaryService:
    """Retrieval-augmented generation: a short explanation written from the retrieved items only."""

    def __init__(self, model_name: str = "gemini-3.5-flash-lite"):
        self.model_name = model_name

    @property
    def available(self) -> bool:
        return bool(os.environ.get("GEMINI_API_KEY"))

    def explain(self, query_text: str, hits) -> Optional[str]:
        if not self.available:
            return None
        try:
            from google import genai

            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            products = "\n".join(
                f"{i}. {h.payload.get('title')} ({h.payload.get('description')}) - similarity {h.score:.2f}"
                for i, h in enumerate(hits, start=1)
            )
            prompt = (
                f"A shopper searched for: {query_text}\n\n"
                f"These are the matching products, best match first:\n{products}\n\n"
                "Write a short, friendly summary (3-4 sentences) of these results. "
                "Mention which one is the closest match and how the others differ. "
                "Use only the products listed above."
            )
            reply = client.models.generate_content(model=self.model_name, contents=prompt)
            return reply.text
        except Exception as e:                       # never let the LLM break a search
            return f"(AI summary unavailable: {e})"


class SearchLogger:
    """SQLite log of every search; used for usage statistics."""

    def __init__(self, path: str):
        self.path = path
        with closing(sqlite3.connect(self.path)) as con, con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS search_log (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       created_at TEXT NOT NULL,
                       mode TEXT NOT NULL,
                       query_text TEXT,
                       colour TEXT,
                       has_image INTEGER NOT NULL,
                       top_ids TEXT,
                       latency_ms REAL NOT NULL,
                       summary_used INTEGER NOT NULL)"""
            )

    def log(self, mode, query_text, colour, has_image, top_ids, latency_ms, summary_used):
        with closing(sqlite3.connect(self.path)) as con, con:
            con.execute(
                "INSERT INTO search_log (created_at, mode, query_text, colour, has_image, top_ids,"
                " latency_ms, summary_used) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), mode, query_text,
                 colour, int(has_image), ",".join(str(i) for i in top_ids), latency_ms,
                 int(summary_used)),
            )

    def stats(self):
        with closing(sqlite3.connect(self.path)) as con:
            con.row_factory = sqlite3.Row
            total = con.execute("SELECT COUNT(*) FROM search_log").fetchone()[0]
            avg = con.execute("SELECT AVG(latency_ms) FROM search_log").fetchone()[0]
            modes = {r["mode"]: r["n"] for r in con.execute(
                "SELECT mode, COUNT(*) AS n FROM search_log GROUP BY mode")}
            recent = [dict(r) for r in con.execute(
                "SELECT created_at, mode, query_text, colour, latency_ms "
                "FROM search_log ORDER BY id DESC LIMIT 10")]
        return {"total_searches": total,
                "average_latency_ms": round(avg, 1) if avg else None,
                "searches_by_mode": modes,
                "recent_searches": recent}


# =============================================================== application
services = {}                       # filled at start-up
lock = threading.Lock()             # CLIP and the embedded Qdrant store serve one request at a time


@asynccontextmanager
async def lifespan(app: FastAPI):
    services["embedder"] = EmbeddingService()
    services["store"] = VectorStore(VECTOR_DB_PATH)
    services["summary"] = SummaryService()
    services["logger"] = SearchLogger(LOG_DB_PATH)
    yield
    services["store"].close()


app = FastAPI(title="Multi-Modal Product Search API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/images", StaticFiles(directory=IMAGE_DIR, check_dir=False), name="images")


def to_product(item, score=None) -> ProductOut:
    p = item.payload
    return ProductOut(
        id=int(item.id), title=p.get("title"), description=p.get("description"),
        colour=p.get("colour"), article_type=p.get("article_type"), gender=p.get("gender"),
        score=None if score is None else round(float(score), 4),
        image_url=f"/images/{os.path.basename(p.get('image_path', f'{item.id}.jpg'))}",
    )


def read_image(upload: UploadFile) -> Image.Image:
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The image is larger than 5 MB.")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")


@app.get("/health")
def health():
    """Is the service up, and how many products are indexed?"""
    store = services["store"]
    return {"status": "ok", "products_indexed": store.count(),
            "embedding_model": services["embedder"].model_name,
            "ai_summary_enabled": services["summary"].available}


@app.get("/colours")
def colours():
    """Colour names present in the index (used by the colour dropdown)."""
    return {"colours": services["store"].colours}


@app.get("/products/{product_id}", response_model=ProductOut)
def product(product_id: int):
    """Metadata of one indexed product."""
    with lock:
        item = services["store"].get(product_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Product {product_id} is not in the index.")
    return to_product(item)


@app.get("/stats")
def stats():
    """Usage statistics from the search log."""
    return services["logger"].stats()


@app.post("/search", response_model=SearchResponse)
def search(
    text: Optional[str] = Form(None),
    colour: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    with_summary: bool = Form(True),
):
    """Search by text, by photo, or by both; optionally restrict to one colour."""
    started = time.perf_counter()

    # 1. validate the request
    text = (text or "").strip() or None
    colour = (colour or "").strip() or None
    if colour and colour.lower() == "any":
        colour = None
    if text and len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail=f"Text is longer than {MAX_TEXT_LENGTH} characters.")
    photo = read_image(image) if (image is not None and image.filename) else None
    if not text and photo is None:
        raise HTTPException(status_code=400, detail="Provide a text description, a photo, or both.")
    store, embedder = services["store"], services["embedder"]
    if colour and colour not in store.colours:
        raise HTTPException(status_code=400, detail=f"Unknown colour '{colour}'. See GET /colours.")
    mode = "hybrid" if (text and photo is not None) else ("text" if text else "image")

    # 2. embed, fuse and search
    with lock:
        vectors = []
        if text:
            vectors.append(embedder.encode_text(text))
        if photo is not None:
            vectors.append(embedder.encode_image(photo))
        query_vector = embedder.fuse(vectors)
        hits = store.search(query_vector, colour=colour, k=TOP_K)
    retrieval_ms = (time.perf_counter() - started) * 1000

    # 3. grounded summary (RAG)
    description = text or "a photo search"
    if colour:
        description += f" (colour filter: {colour})"
    summary = services["summary"].explain(description, hits) if (with_summary and hits) else None

    # 4. log and respond
    latency_ms = (time.perf_counter() - started) * 1000
    services["logger"].log(mode, text, colour, photo is not None, [h.id for h in hits],
                           round(latency_ms, 1), bool(summary))
    return SearchResponse(
        mode=mode, colour_filter=colour, retrieval_ms=round(retrieval_ms, 1),
        latency_ms=round(latency_ms, 1), summary=summary,
        results=[to_product(h, h.score) for h in hits],
    )
