
# Multi-Modal Product Search (CLIP + Qdrant + RAG)

Search a fashion catalog with **text, a photo, or both**. Product images are embedded with CLIP,
stored in the Qdrant vector database, and the top results are explained by an LLM (Gemini) using
only the retrieved products (retrieval-augmented generation). An optional colour filter handles
requests like "similar to this photo, but in blue".

## Architecture

- **Frontend:** Streamlit (`app.py`), search panel on the left, results on the right.
- **Backend:** FastAPI (`backend.py`) with four services: `EmbeddingService` (CLIP),
  `VectorStore` (Qdrant), `SummaryService` (Gemini), `SearchLogger` (SQLite).
- **Indexing:** `build_index.py` embeds product images and writes them to Qdrant.
- **Evaluation:** `evaluate.py` measures precision@5 and latency for four retrieval modes.

API endpoints: `POST /search`, `GET /health`, `GET /colours`, `GET /products/{id}`, `GET /stats`.
Interactive docs: `http://127.0.0.1:8000/docs`.

## Setup (Windows PowerShell, Python 3.13)

1. Install the packages:
   ```
   pip install -r requirements.txt
   ```
2. Download the **Fashion Product Images (Small)** dataset from Kaggle
   (https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) and put it in
   the project folder like this:
   ```
   data/styles.csv
   data/images/<id>.jpg
   ```
   The dataset is not included in this repository.
3. Build the search index (about 5 to 10 minutes; creates `~/qdrant_store`):
   ```
   python build_index.py
   ```
4. Optional, for the AI summary, set a free Gemini API key from Google AI Studio:
   ```
   $env:GEMINI_API_KEY="your-key"
   ```
5. Start the backend, then the frontend in a second terminal:
   ```
   uvicorn backend:app --port 8000
   streamlit run app.py
   ```

Only one program can open the Qdrant folder at a time, so stop the backend before running
`build_index.py` or `evaluate.py`.

## Results (50 queries per mode, 2,000 indexed products, CPU only)

| Mode | P@5 (%) | Latency (ms) |
|---|---|---|
| Text only | 50.0 | 62 |
| Image only | 78.8 | 446 |
| Photo + text (fused) | 28.0 | 553 |
| Photo + colour filter | 76.4 | 510 |

Modes 1, 3 and 4 need article type and colour to match; mode 2 needs article type only.

## Known limitations

- Retrieval is closed-world: a query for a product that is not in the catalog returns the
  nearest items instead of "no match".
- Averaging a photo vector with a text vector does not reliably change an attribute such as colour;
  the colour filter is used for that.
- The embedded Qdrant store and the single-request lock are meant for a prototype.

## Authors

Add your names here.

# Multimodal-Project-Search-using-RAG
e9ae0ae41833890ad8f9a98f6b3d9b2d019fa6bc
