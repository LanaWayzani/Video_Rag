import os
import numpy as np
import json
import psycopg2

# --- Config ---
DB_NAME = "videoqa"
USER = "postgres"
PASSWORD = "Tuti1311"
HOST = "localhost"
PORT = 5433

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EMB_PATH = os.path.join(ROOT, "data", "embeddings", "image_embeddings.npy")
META_PATH = os.path.join(ROOT, "data", "keyframes", "metadata.json")

# --- Load data ---
embeddings = np.load(EMB_PATH).astype("float32")
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

assert len(embeddings) == len(metadata), "Mismatch in image embeddings and metadata!"

# --- Connect ---
conn = psycopg2.connect(
    dbname=DB_NAME, user=USER, password=PASSWORD, host=HOST, port=PORT
)
cur = conn.cursor()

cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

cur.execute("""
DROP TABLE IF EXISTS keyframes;
CREATE TABLE keyframes (
    id SERIAL PRIMARY KEY,
    timestamp FLOAT,
    image_path TEXT,
    embedding vector(512)
);
""")

for meta, emb in zip(metadata, embeddings):
    cur.execute(
        "INSERT INTO keyframes (timestamp, image_path, embedding) VALUES (%s, %s, %s);",
        (meta.get("time") or meta.get("timestamp", -1), meta["image_path"], emb.tolist())
    )

conn.commit()

# --- Index
cur.execute("""
CREATE INDEX keyframe_embedding_idx
ON keyframes USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
""")

conn.commit()
cur.close()
conn.close()

print("✅ Keyframe embeddings inserted and indexed (IVFFLAT).")
