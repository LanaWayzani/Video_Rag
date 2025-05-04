import os
import numpy as np
import json
import psycopg2
from dotenv import load_dotenv

#Load environment variables from .env file
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


#Just to ensure the embeddings and segments are consistent
#assert len(embeddings) == len(segments), "Mismatch in text embeddings and segments"

#Connecting to PostgreSQL database
conn = psycopg2.connect(dbname=data_base, user=USER, password=PASSWORD, host=HOST, port=PORT)
cur = conn.cursor()

cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# Create table for storing transcript segments and their vector embeddings
cur.execute("""DROP TABLE IF EXISTS transcript_segments_hnsw;
CREATE TABLE transcript_segments_hnsw (
    id SERIAL PRIMARY KEY,
    start FLOAT,
    text TEXT,
    embedding vector(544));
    """)

#Insert transcript text segments and corresponding compressed embeddings into the table
for seg, emb in zip(segments, embeddings):
    cur.execute("INSERT INTO transcript_segments_hnsw (start, text, embedding) VALUES (%s, %s, %s);",
        (seg["start"], seg["text"], emb.tolist()))

conn.commit()

# Create HNSW index on the embedding column for fast approximate nearest neighbor search
cur.execute("""DROP INDEX IF EXISTS transcript_embedding_hnsw;
CREATE INDEX transcript_embedding_hnsw
ON transcript_segments_hnsw USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);""")
# The following are tunable parameters:m = 16 is the number of bi-directional links created for each new element 
# and ef_construction = 64 is the size of the dynamic list for the nearest neighbors during index construction.

conn.commit()
cur.close()
conn.close()

print("Transcript segments inserted and indexed.")
