import os
import numpy as np
import psycopg2
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ----------------------------
# CONFIG
# ----------------------------
DB_NAME = "videoqa"
USER = "postgres"
PASSWORD = "your_password"
HOST = "localhost"
PORT = 5433
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Load CLIP
# ----------------------------
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE).eval()
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

def embed_query_clip(text):
    inputs = clip_processor(text=[text], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = clip_model.get_text_features(**inputs)
        vec = output.squeeze().cpu().numpy()
        vec = vec / np.linalg.norm(vec)
        return vec

# ----------------------------
# Query PostgreSQL (pgvector)
# ----------------------------
def search_images_pgvector(query_vec, k=3):
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT
    )
    cur = conn.cursor()

    query_vec_str = "[" + ",".join(map(str, query_vec.tolist())) + "]"

    cur.execute(f"""
        SELECT timestamp, image_path, embedding <-> %s AS similarity
        FROM keyframes
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
    print("Enter your visual question about the video:")
    query = input("> ")

    query_vec = embed_query_clip(query)
    results = search_images_pgvector(query_vec)

    print("\n--- Top Keyframe Matches (pgvector IVFFLAT) ---")
    for timestamp, path, sim in results:
        print(f"[{timestamp:.2f}s] {path}  [score={1 - sim:.4f}]")
