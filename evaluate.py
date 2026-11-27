"""
evaluate.py - measures precision@5 and latency for the four retrieval modes
used in the paper (Table II).

HOW TO RUN (from your project folder, with the app and build_index.py CLOSED):
    python evaluate.py

It needs:
  * data/styles.csv and data/images/ (same as build_index.py)
  * the Qdrant database built by build_index.py, with the payload fields
    "colour" and "article_type" (added in the colour-filter step).
It does NOT use the Gemini API, so it costs nothing and needs no key.
"""
import os
import time
import random

import numpy as np
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

N_QUERIES = 50      # queries per mode (change if you want more or fewer)
K = 5               # precision@K
MIN_RELEVANT = 5    # a target must have at least this many matching products in the index
SEED = 42
DB_PATH = os.path.join(os.path.expanduser("~"), "qdrant_store")   # same as build_index.py / app.py

rng = random.Random(SEED)
model = SentenceTransformer("clip-ViT-B-32")
client = QdrantClient(path=DB_PATH)


def normalize(v):
    return v / np.linalg.norm(v)


def search(vec, colour=None):
    flt = None
    if colour:
        flt = Filter(must=[FieldCondition(key="colour", match=MatchValue(value=colour))])
    return client.query_points("products", query=vec.tolist(), query_filter=flt, limit=K).points


def precision(hits, want_type, want_colour=None):
    rel = 0
    for h in hits:
        ok = h.payload.get("article_type") == want_type
        if want_colour is not None:
            ok = ok and h.payload.get("colour") == want_colour
        rel += int(ok)
    return rel / K


# ------------------------------------------------------------------ 1. what is in the index?
indexed = {}                      # product id -> (article_type, colour)
offset = None
while True:
    points, offset = client.scroll("products", limit=500, offset=offset,
                                   with_payload=True, with_vectors=False)
    for p in points:
        indexed[p.id] = (p.payload.get("article_type"), p.payload.get("colour"))
    if offset is None:
        break

if not indexed or all(t is None for t, _ in indexed.values()):
    raise SystemExit("The index has no 'article_type' / 'colour' fields. Re-run build_index.py "
                     "after adding them (colour-filter step), then run this script again.")

pair_count = {}                   # (article_type, colour) -> number of indexed products
type_count = {}                   # article_type -> number of indexed products
for t, c in indexed.values():
    if t is None or c is None:
        continue
    pair_count[(t, c)] = pair_count.get((t, c), 0) + 1
    type_count[t] = type_count.get(t, 0) + 1
print(f"Index holds {len(indexed)} products.")

# ------------------------------------------------------------------ 2. query photos (NOT in the index)
df = pd.read_csv("data/styles.csv", on_bad_lines="skip")
df = df.dropna(subset=["articleType", "baseColour"])
df["image_path"] = df["id"].astype(str).apply(lambda i: f"data/images/{i}.jpg")
df = df[df["image_path"].apply(os.path.exists)]
df = df[~df["id"].isin(set(indexed.keys()))]
df = df[df["articleType"].map(lambda t: type_count.get(t, 0) >= MIN_RELEVANT)]
if df.empty:
    raise SystemExit("No catalog photos are left outside the index to use as queries. "
                     "Index only a sample (e.g. 2000 products) so query photos are not in the index.")
rows = df.to_dict("records")
rng.shuffle(rows)

# ------------------------------------------------------------------ 3. build the queries
# Mode 1 (text only): "<colour> <type>", target = that type AND colour
pairs = [pc for pc, n in pair_count.items() if n >= MIN_RELEVANT]
rng.shuffle(pairs)
text_queries = pairs[:N_QUERIES]

# Mode 2 (image only): target = same article type as the photo
image_queries = rows[:N_QUERIES]

# Modes 3 and 4 (photo + other colour): target = same type, requested colour
colour_queries = []
for r in rows:
    options = [c for (t, c), n in pair_count.items()
               if t == r["articleType"] and c != r["baseColour"] and n >= MIN_RELEVANT]
    if options:
        colour_queries.append((r, rng.choice(sorted(options))))
    if len(colour_queries) == N_QUERIES:
        break

# ------------------------------------------------------------------ 4. warm-up (so timing excludes start-up)
model.encode("warm up")
warm = Image.open(rows[0]["image_path"]).convert("RGB")
search(normalize(model.encode(warm)))

# ------------------------------------------------------------------ 5. run the four modes
results = {1: [], 2: [], 3: [], 4: []}   # mode -> list of (description, precision, latency_ms)

for (t, c) in text_queries:                                   # mode 1
    q = f"{c.lower()} {t.lower()}"
    start = time.perf_counter()
    vec = normalize(model.encode(q))
    hits = search(vec)
    ms = (time.perf_counter() - start) * 1000
    results[1].append((q, precision(hits, t, c), ms))

for r in image_queries:                                       # mode 2
    img = Image.open(r["image_path"]).convert("RGB")
    start = time.perf_counter()
    vec = normalize(model.encode(img))
    hits = search(vec)
    ms = (time.perf_counter() - start) * 1000
    results[2].append((f"photo {r['id']} ({r['articleType']})", precision(hits, r["articleType"]), ms))

for r, c in colour_queries:                                   # modes 3 and 4
    img = Image.open(r["image_path"]).convert("RGB")
    desc = f"photo {r['id']} ({r['articleType']}) -> {c}"

    start = time.perf_counter()                               # mode 3: photo + colour word, fused
    vec = normalize(normalize(model.encode(img)) + normalize(model.encode(c.lower())))
    hits = search(vec)
    ms = (time.perf_counter() - start) * 1000
    results[3].append((desc, precision(hits, r["articleType"], c), ms))

    start = time.perf_counter()                               # mode 4: photo + colour filter
    vec = normalize(model.encode(img))
    hits = search(vec, colour=c)
    ms = (time.perf_counter() - start) * 1000
    results[4].append((desc, precision(hits, r["articleType"], c), ms))

# ------------------------------------------------------------------ 6. report
names = {1: "1. Text only", 2: "2. Image only",
         3: "3. Photo + text (fused)", 4: "4. Photo + colour filter"}
print("\n=== RESULTS (copy these into Table II) ===")
print(f"{'Mode':28s} {'Queries':>8s} {'P@5 (%)':>9s} {'Latency (ms)':>13s}")
rows_out = []
for m in (1, 2, 3, 4):
    data = results[m]
    n = len(data)
    p5 = 100 * np.mean([d[1] for d in data]) if n else float("nan")
    lat = np.mean([d[2] for d in data]) if n else float("nan")
    print(f"{names[m]:28s} {n:8d} {p5:9.1f} {lat:13.0f}")
    for d in data:
        rows_out.append({"mode": names[m], "query": d[0], "precision_at_5": d[1], "latency_ms": round(d[2], 1)})

pd.DataFrame(rows_out).to_csv("eval_per_query.csv", index=False)
print("\nPer-query details saved to eval_per_query.csv")
print("If the 'Queries' counts differ between modes, write the smallest count as N in the paper "
      "or report each count.")