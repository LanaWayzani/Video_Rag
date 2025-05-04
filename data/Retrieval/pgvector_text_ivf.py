import psycopg2
import numpy as np
import json
import os
from dotenv import load_dotenv

load_dotenv()  
data_base = os.getenv("data_base")
USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
PORT = int(os.getenv("PORT"))

#Defining paths for embeddings and transcript segments
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
embed_path = os.path.join(root, "data", "embeddings", "compressed_text_embeddings.npy")
seg_path = os.path.join(root, "data", "transcripts", "segments.json")
embeddings = np.load(embed_path).astype("float32")
with open(seg_path, "r", encoding="utf-8") as f:
    segments = json.load(f)


# assert len(embeddings) == len(segments), "Mismatch in embeddings and segments"

conn = psycopg2.connect(dbname=data_base, user=USER, password=PASSWORD, host=HOST, port=PORT)
cur = conn.cursor()
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

#Creating a table for transcript segments with vector embeddings
cur.execute("""
DROP TABLE IF EXISTS transcript_segments;
CREATE TABLE transcript_segments (
    id SERIAL PRIMARY KEY,
    text TEXT,
    start FLOAT,
    embedding vector(544));""")

#Inserting each segment along with its embedding
for i, (seg, emb) in enumerate(zip(segments, embeddings)):
    cur.execute( "INSERT INTO transcript_segments (text, start, embedding) VALUES (%s, %s, %s);",
        (seg["text"], seg["start"], emb.tolist()))

conn.commit()

#Creating an IVFFLAT index on the embedding column for fast approximate nearest neighbor search
cur.execute("""CREATE INDEX transcript_embedding_idx
ON transcript_segments USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
""")

conn.commit()
cur.close()
conn.close()

print("Embeddings inserted and Ivfflat index created.")
