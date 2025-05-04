import os
import numpy as np
import json
import psycopg2
from dotenv import load_dotenv


load_dotenv()  
data_base = os.getenv("data_base")
USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
PORT = int(os.getenv("PORT"))

#Defining paths for image embeddings and corresponding metadata
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
embed_path = os.path.join(root, "data", "embeddings", "image_embeddings.npy")
meta_path = os.path.join(root, "data", "keyframes", "metadata.json")


embeddings = np.load(embed_path).astype("float32")
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

with open(meta_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

#Ensuring matching lengths
#assert len(embeddings) == len(metadata), "Mismatch in image embeddings and metadata"


conn = psycopg2.connect(dbname=data_base, user=USER, password=PASSWORD, host=HOST, port=PORT)
cur = conn.cursor()
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

#creating a table for keyframes with vector embeddings and metadata
cur.execute("""DROP TABLE IF EXISTS keyframes;
CREATE TABLE keyframes (
    id SERIAL PRIMARY KEY,
    timestamp FLOAT,
    image_path TEXT,
    embedding vector(512));""")

#Inserting each keyframe along with its embedding
for meta, emb in zip(metadata, embeddings):
    cur.execute("INSERT INTO keyframes (timestamp, image_path, embedding) VALUES (%s, %s, %s);",
    (meta.get("time") or meta.get("timestamp", -1), meta["image_path"], emb.tolist()))

conn.commit()

#Creating an IVFFLAT index on the vector column for fast approximate similarity search
cur.execute("""CREATE INDEX keyframe_embedding_idx
ON keyframes USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);""")
# The parameter lists = 100 is tunable for performance optimization.It indicates the centroids used to partition the embedding space. 

conn.commit()
cur.close()
conn.close()

print("Keyframe embeddings inserted and indexed.")
