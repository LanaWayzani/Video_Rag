import psycopg2
import numpy as np
import json
import os

# --- Config ---
DB_NAME = "videoqa"
USER = "postgres"
PASSWORD = "Tuti1311"
HOST = "localhost"
PORT = 5432

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EMB_PATH = os.path.join(ROOT, "data", "embeddings", "text_embeddings.npy")
SEGMENT_PATH = os.path.join(ROOT, "data", "transcripts", "segments.json")

# --- Load data ---
embeddings = np.load(EMB_PATH).astype("float32")
with open(SEGMENT_PATH, "r", encoding="utf-8") as f:
    segments = json.load(f)

assert len(embeddings) == len(segments), "Mismatch in embeddings and segments count!"

# --- Connect & Create Table ---
conn = psycopg2.connect(
    dbname=DB_NAME, user=USER, password=PASSWORD, host=HOST, port=PORT
)
cur = conn.cursor()

cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

cur.execute("""
DROP TABLE IF EXISTS transcript_segments;
CREATE TABLE transcript_segments (
    id SERIAL PRIMARY KEY,
    text TEXT,
    start FLOAT,
    embedding vector(1024)
);
""")

# --- Insert Data ---
for i, (seg, emb) in enumerate(zip(segments, embeddings)):
    cur.execute(
        "INSERT INTO transcript_segments (text, start, embedding) VALUES (%s, %s, %s);",
        (seg["text"], seg["start"], emb.tolist())
    )

conn.commit()

# --- Create IVFFLAT index ---
cur.execute("""
CREATE INDEX transcript_embedding_idx
ON transcript_segments USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
""")

conn.commit()
cur.close()
conn.close()

print("✅ Embeddings inserted and IVFFLAT index created.")
