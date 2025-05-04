import os
import json
import numpy as np
import psycopg2
import torch
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
from PIL import Image
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
DB_NAME = "videoqa"
USER = "postgres"
PASSWORD = "Tuti1311"
HOST = "localhost"
PORT = 5433

E5_MODEL = "intfloat/e5-large-v2"
CLIP_MODEL = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIM_THRESHOLD = 0.3

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ----------------------------
# Load Models
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(E5_MODEL)
e5_model = AutoModel.from_pretrained(E5_MODEL).to(DEVICE).eval()
clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE).eval()
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

# ----------------------------
# Embedding Functions
# ----------------------------
def embed_text(query):
    inputs = tokenizer(f"query: {query}", return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        out = e5_model(**inputs)
        vec = out.last_hidden_state.mean(dim=1)
        vec = torch.nn.functional.normalize(vec, p=2, dim=1)
        return vec.squeeze().cpu().numpy()

def embed_image_query(text):
    inputs = clip_processor(text=[text], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = clip_model.get_text_features(**inputs)
    vec = out.squeeze().cpu().numpy()
    return vec / np.linalg.norm(vec)

# ----------------------------
# Search Functions (HNSW)
# ----------------------------
def search_text_pgvector(query_vec, k=1):
    conn = psycopg2.connect(dbname=DB_NAME, user=USER, password=PASSWORD, host=HOST, port=PORT)
    cur = conn.cursor()
    vec_str = "[" + ",".join(map(str, query_vec.tolist())) + "]"
    cur.execute(f"""
        SELECT text, start, embedding <-> %s AS similarity
        FROM transcript_segments_hnsw
        ORDER BY similarity ASC
        LIMIT {k};
    """, (vec_str,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result

def search_image_pgvector(query_vec, k=1):
    conn = psycopg2.connect(dbname=DB_NAME, user=USER, password=PASSWORD, host=HOST, port=PORT)
    cur = conn.cursor()
    vec_str = "[" + ",".join(map(str, query_vec.tolist())) + "]"
    cur.execute(f"""
        SELECT timestamp, image_path, embedding <-> %s AS similarity
        FROM keyframes_hnsw
        ORDER BY similarity ASC
        LIMIT {k};
    """, (vec_str,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    print("Ask your question about the video:")
    query = input("> ")

    text_vec = embed_text(query)
    image_vec = embed_image_query(query)

    text_result = search_text_pgvector(text_vec)
    image_result = search_image_pgvector(image_vec)

    text_score = 1 - text_result[2] if text_result else 0
    image_score = 1 - image_result[2] if image_result else 0

    print("\n--- Top Matches (HNSW) ---")

    print(f"\n Transcript Match:")
    print(f"[{text_result[1]:.2f}s] {text_result[0]}  [score={text_score:.4f}]")

    print(f"\n Image Match:")
    print(f"[{image_result[0]:.2f}s] {image_result[1]}  [score={image_score:.4f}]")

    image_path = os.path.join(ROOT, image_result[1])
    if os.path.exists(image_path):
        img = Image.open(image_path)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{image_result[0]:.1f}s  |  score={image_score:.4f}")
        plt.show()
    else:
        print(" Image not found.")
