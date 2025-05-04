import os
import json
import psycopg2
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# ----------------------------
# CONFIG
# ----------------------------
DB_NAME = "videoqa"
USER = "postgres"
PASSWORD = "Tuti1311"
HOST = "localhost"
PORT = 5433  # <-- using PostgreSQL 15 port
TEXT_MODEL_NAME = "intfloat/e5-large-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Embed Query using E5
# ----------------------------
def embed_query(text, model, tokenizer):
    prompt = f"query: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        output = model(**inputs)
        emb = output.last_hidden_state.mean(dim=1)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.squeeze().cpu().numpy()

# ----------------------------
# Connect and Query pgvector
# ----------------------------
def search_transcripts_pgvector(query_vec, k=5):
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT
    )
    cur = conn.cursor()

    # Convert numpy vector to SQL-friendly string
    query_vec_str = "[" + ",".join(map(str, query_vec.tolist())) + "]"

    cur.execute(f"""
        SELECT text, start, embedding <-> %s AS similarity
        FROM transcript_segments
        ORDER BY similarity ASC
        LIMIT {k};
    """, (query_vec_str,))

    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    model = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(DEVICE).eval()

    print("Enter your question about the video:")
    query = input("> ")

    query_vec = embed_query(query, model, tokenizer)
    results = search_transcripts_pgvector(query_vec)

    print("\n--- Top Transcript Matches (pgvector IVFFLAT) ---")
    for text, start, similarity in results:
        print(f"[{start:.2f}s] {text}  [score={1 - similarity:.4f}]")
