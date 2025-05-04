import os
import json
import numpy as np
import psycopg2
import torch
from dotenv import load_dotenv
import joblib

"""
Throughout this code, we clarify the building and querying HNSW indexes with pgvector for transcript and image embeddings.

This module:
1. Loads compressed text and image embeddings
2.  Inserts them into a PostgreSQL database using pgvector with HNSW indexing
3.  Defines embedding and search functions
4.  Supports multimodal fusion retrieval
"""

# we load environment variables from a .env file
load_dotenv()
data_base = os.getenv("data_base")
USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")


#Defining paths to data directories
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
embed_path = os.path.join(root, "data", "embeddings", "compressed_text_embeddings.npy")
comb_text_pca_path = os.path.join(root, "data", "embeddings", "combined_text_embeddings_pca.npy")
comb_imag_path = os.path.join(root, "data", "embeddings", "combined_image_embeddings.npy")
comb_meta_path = os.path.join(root, "data", "embeddings", "combined_metadata_aligned.json")
seg_path = os.path.join(root, "data", "transcripts", "segments.json")


# Load compressed text and image embeddings and normalize them
embeddings = np.load(embed_path).astype("float32")
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
combined_image = np.load(comb_imag_path).astype("float32")
combined_image /= np.linalg.norm(combined_image, axis=1, keepdims=True)

with open(comb_meta_path, "r", encoding="utf-8") as f:
    combined_meta = json.load(f)

with open(seg_path, "r", encoding="utf-8") as f:
    segments = json.load(f)

"""
# Ensure consistency between embeddings and metadata
assert combined_image.shape[0] == len(combined_meta), "Mismatch: combined_image and combined_meta"
assert len(embeddings) == len(segments), "Mismatch in embeddings and segments count!"
"""

Text_Threshold = 0.50
Image_Threshold = 0.38

#Creating a hnsw index for the text embeddings in PostgreSQL
conn = psycopg2.connect(dbname=data_base, user=USER, password=PASSWORD, host=HOST, port=PORT)
cur = conn.cursor()

cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

#Drop the table if it exists and create a new one for transcript segments
cur.execute("DROP TABLE IF EXISTS transcript_segments_hnsw;")
cur.execute("""CREATE TABLE transcript_segments_hnsw (
        id SERIAL PRIMARY KEY,
        start FLOAT,
        text TEXT,
        embedding vector(544));
            """)

# Insert data row by row
for seg, emb in zip(segments, embeddings):
    cur.execute("INSERT INTO transcript_segments_hnsw (start, text, embedding) VALUES (%s, %s, %s);",
        (seg["start"], seg["text"], emb.tolist())
    )

conn.commit()

# Build HNSW index for approximate nearest neighbor search where we used m=16 (which controls the number of bi-directional links created per node) and 
# ef_construction=64 ( the size of the candidate list during index construction).
# These parameters can be tuned for performance vs. memory trade-offs.
cur.execute("""
    CREATE INDEX transcript_embedding_hnsw
    ON transcript_segments_hnsw USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
""")
conn.commit()
cur.close()
conn.close()

print("Transcript compressed embeddings inserted and HNSW index created.")

pca = joblib.load("Video_Rag/data/embeddings/pca_model.pkl")

def embed_text_query(text, tokenizer, text_model):
    """
    Embed a query string using the SFR model and compress using fitted PCA.
    Returns a normalized 544-dimensional ( based on pca) vector for PostgreSQL search.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    device = next(text_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = text_model(**inputs)
        emb = output.last_hidden_state.mean(dim=1)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1).cpu().numpy()

    compressed = pca.transform(emb)
    compressed /= np.linalg.norm(compressed, axis=1, keepdims=True)
    return compressed.squeeze()

def embed_clip_query(text, clip_processor, clip_model):
    """
    Embed a query string using CLIP's text encoder.
    Returns a normalized 512-dimensional vector.
    """
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    device = next(clip_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = clip_model.get_text_features(**inputs)
        output = torch.nn.functional.normalize(output, p=2, dim=1)
    return output.squeeze().cpu().numpy()



def search_text_pgvector_hnsw(query_vec, k=5):
    """
    Query pgvector HNSW index for top-k transcript matches based on cosine similarity.
    """
    conn = psycopg2.connect(dbname=data_base, user=USER, password=PASSWORD, host=HOST, port=PORT)
    cur = conn.cursor()

    vec_str = "[" + ",".join(map(str, query_vec.tolist())) + "]"
    cur.execute(f"""SELECT id, start, 1 - (embedding <-> %s) / 2.0 AS similarity, text
        FROM transcript_segments_hnsw
        ORDER BY similarity DESC
        LIMIT {k};""", (vec_str,))

    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [(row[0], row[1], row[3], round(row[2], 4)) for row in rows]

def search_image_pgvector_hnsw(query_vec, k=5):
    """
    Query pgvector HNSW index for top-k keyframe image matches..
    """
    conn = psycopg2.connect(dbname=data_base, user=USER, password=PASSWORD, host=HOST, port=PORT)
    cur = conn.cursor()

    vec_str = "[" + ",".join(map(str, query_vec.tolist())) + "]"
    cur.execute(f"""SELECT id, timestamp, image_path, 1 - (embedding <-> %s) / 2.0 AS similarity
        FROM keyframes_hnsw
        ORDER BY similarity DESC
        LIMIT {k};""", (vec_str,))

    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [(row[1], row[2], round(float(row[3]), 4)) for row in rows]




def run_pgvector_hnsw_multimodal(query, k, tokenizer, text_model, clip_processor, clip_model, alpha=0.5):
    """
    Perform fusion search combining text and image embeddings.
    Scores are normalized and combined through weighted sum with alpha.
    Returns transcript matches, image matches, and fused results.
    """
    text_vec = embed_text_query(query, tokenizer, text_model)
    image_vec = embed_clip_query(query, clip_processor, clip_model)

    combined_text = np.load(comb_text_pca_path).astype("float32")
    combined_text /= np.linalg.norm(combined_text, axis=1, keepdims=True)

    text_scores = np.dot(combined_text, text_vec)
    image_scores = np.dot(combined_image, image_vec)

    text_scores /= (np.max(text_scores) + 1e-9)
    image_scores /= (np.max(image_scores) + 1e-9)

    fused_scores = alpha * text_scores + (1 - alpha) * image_scores
    top_indices = np.argsort(fused_scores)[::-1][:k]

    transcript_results = [(combined_meta[i]["timestamp"], combined_meta[i]["text"], round(text_scores[i], 4)) for i in top_indices]
    image_results = [(combined_meta[i]["timestamp"], combined_meta[i]["image_path"], round(image_scores[i], 4)) for i in top_indices]
    fused = [{"timestamp": combined_meta[i]["timestamp"],
        "text": combined_meta[i]["text"],
        "image_path": combined_meta[i]["image_path"],
        "fused_score": round(fused_scores[i], 4)
    } for i in top_indices]

    return transcript_results, image_results, fused

def run_pgvector_hnsw(query, k, tokenizer, text_model, clip_processor, clip_model, mode="Text", alpha=0.5):
    if mode == "Text":
        text_vec = embed_text_query(query, tokenizer, text_model)
        text_vec /= np.linalg.norm(text_vec) + 1e-9
        scores = np.dot(embeddings, text_vec)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(segments[i]["start"], segments[i]["text"], round(scores[i], 4)) for i in top_indices], [], []

    elif mode == "Image":
        image_vec = embed_clip_query(query, clip_processor, clip_model)
        return [], search_image_pgvector_hnsw(image_vec, k), []

    elif mode == "Multimodal Fusion":
        return run_pgvector_hnsw_multimodal(query, k, tokenizer, text_model, clip_processor, clip_model, alpha)

    return [], [], []


print("HNSW fusion retrieval done")

"""
def main():
    from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel

    Text_model = "Salesforce/SFR-Embedding-Mistral"
    clip_model_image = "openai/clip-vit-base-patch32"

    tokenizer = AutoTokenizer.from_pretrained(Text_model)
    text_model = AutoModel.from_pretrained(Text_model).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_image)
    clip_model = CLIPModel.from_pretrained(clip_model_image).eval()

    print("Type 'exit' to quit.\n")

    while True:
        query = input("Enter a query: ").strip()
        if query.lower() == "exit":
            break

        mode = input("Select mode [Text / Image / Multimodal Fusion]: ").strip().lower()
        if mode == "exit":
            break
        if mode not in {"text", "image", "multimodal fusion"}:
            print("Invalid mode. Please try again.\n")
            continue

        alpha = 0.5
        if mode == "multimodal fusion":
            try:
                alpha = float(input("Enter fusion weight alpha (0–1): ").strip())
            except ValueError:
                print("Default to 0.5.")

        try:
            k = int(input("How many top results? (k): ").strip())
        except ValueError:
            print("Default to 5.")
            k = 5

        transcript_results, image_results, fused_matches = run_pgvector_hnsw(
            query, k, tokenizer, text_model, clip_processor, clip_model, mode=mode.title(), alpha=alpha)

        print("\n--- Top Results ---")
        if mode == "text":
            if transcript_results and transcript_results[0][2] >= Text_Threshold:
                for ts, text, score in transcript_results:
                    print(f"[Text] ({ts:.2f}s): {text} (score: {score:.3f})")
            else:
                print("No relevant text results (below threshold or empty).")

        elif mode == "image":
            if image_results and image_results[0][2] >= Image_Threshold:
                for ts, path, score in image_results:
                    print(f"[Image] ({ts:.2f}s): {path} (score: {score:.3f})")
            else:
                print("No relevant image results (below threshold or empty).")

        elif mode == "multimodal fusion":
            if fused_matches:
                for item in fused_matches:
                    print(f"[Fusion] ({item['timestamp']:.2f}s): {item['text']} -> {item['image_path']} (score: {item['fused_score']:.3f})")
            else:
                print("No fused matches found.")

        print("-" * 40 + "\n")

if __name__ == "__main__":
    main()
"""
