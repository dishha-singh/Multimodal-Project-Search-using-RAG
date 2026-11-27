import os
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

N_PRODUCTS = 2000                                            # how many products to index
DB_PATH = os.path.join(os.path.expanduser("~"), "qdrant_store")

# 1. Load the product list (skip a few broken rows)
df = pd.read_csv("data/styles.csv", on_bad_lines="skip")
df = df.dropna(subset=["productDisplayName"])

# 2. Keep only products whose image file exists
df["image_path"] = df["id"].astype(str).apply(lambda i: f"data/images/{i}.jpg")
df = df[df["image_path"].apply(os.path.exists)]

# 3. Use a random sample of products
df = df.sample(n=min(N_PRODUCTS, len(df)), random_state=42)


def clean(x):
    return None if pd.isna(x) else str(x)


# 4. Make a short text description for each product
def make_description(row):
    bits = [row["baseColour"], row["articleType"], row["gender"], row["season"], row["usage"]]
    return ", ".join(str(b) for b in bits if pd.notna(b))


# 5. Load CLIP and create the database
model = SentenceTransformer("clip-ViT-B-32")
client = QdrantClient(path=DB_PATH)

if client.collection_exists("products"):
    client.delete_collection("products")
client.create_collection(
    collection_name="products",
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)

# 6. Turn every product photo into a fingerprint and store it
points = []
total = len(df)
for n, (_, row) in enumerate(df.iterrows(), start=1):
    try:
        image = Image.open(row["image_path"]).convert("RGB")
    except Exception:
        continue  # skip any broken image
    vector = model.encode(image)
    points.append(
        PointStruct(
            id=int(row["id"]),
            vector=vector.tolist(),
            payload={
                "title": row["productDisplayName"],
                "description": make_description(row),
                "image_path": row["image_path"],
                "colour": clean(row["baseColour"]),
                "article_type": clean(row["articleType"]),
                "gender": clean(row["gender"]),
            },
        )
    )
    if len(points) == 500:
        client.upsert(collection_name="products", points=points)
        points = []
    if n % 100 == 0:
        print(f"Done {n} of {total}")

if points:
    client.upsert(collection_name="products", points=points)
print("Finished! Your search index is ready.")