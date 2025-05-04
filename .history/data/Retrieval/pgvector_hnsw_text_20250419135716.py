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
EMB_PATH = os.path.join(ROOT, "data", "embeddings", "text_embeddings.npy")
SEGMENT_PATH = os.path.join(ROOT, "data", "transcripts", "segments.json")

# --- Load data ---
embeddings = np.load(EMB_PATH).astype("float32")
with open(SEGMENT_PATH, "r", encoding="utf-8") as f:
    segments = json.load(f)

assert len(embeddings) == len(segments), "Mismatch in text embeddings and segments!"

# --- Connect ---
conn = psycopg2.connect(
    dbname=DB_NAME, user=USER, password=PASSWORD, host=HOST, port=PORT
)
cur = conn.cursor()

cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# --- Create HNSW Table ---
cur.execute("""
DROP TABLE IF EXISTS transcript_segments_hnsw;
CREATE TABLE transcript_segments_hnsw (
    id SERIAL PRIMARY KEY,
    start FLOAT,
    text TEXT,
    embedding vector(1024)
);
""")

# --- Insert segments and embeddings ---
for seg, emb in zip(segments, embeddings):
    cur.execute(
        "INSERT INTO transcript_segments_hnsw (start, text, embedding) VALUES (%s, %s, %s);",
        (seg["start"], seg["text"], emb.tolist())
    )

conn.commit()

# --- Create HNSW Index ---
cur.execute("""
CREATE INDEX transcript_embedding_hnsw
ON transcript_segments_hnsw USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
""")

conn.commit()
cur.close()
conn.close()

print("✅ Transcript segments (HNSW) inserted and indexed.")
