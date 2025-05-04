import os
import json
import numpy as np
import psycopg2
import torch
from dotenv import load_dotenv
import joblib

"""
This script handles retrieval using PostgreSQL with the pgvector extension and IVFFLAT indexing. It supports three modes of querying:
1. Text-based search (compressed transcript embeddings)
2. Image-based search (CLIP embeddings from keyframes)
3. Multimodal fusion ( fusion of text and image similarities)

It embeds queries using pre-trained models, compares against PCA-compressed and normalized vectors,
and retrieves the top-k most relevant entries from pgvector-indexed tables.

"""

# Load environment variables
load_dotenv()
data_base = os.getenv("data_base")
USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
PORT = int(os.getenv("PORT"))

#Defining paths to data directories
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
embed_path = os.path.join(root, "data", "embeddings", "compressed_text_embeddings.npy")
seg_path = os.path.join(root, "data", "transcripts", "segments.json")
comb_meta_path = os.path.join(root, "data", "embeddings", "combined_metadata_aligned.json")
comb_text_pca_path = os.path.join(root, "data", "embeddings", "combined_text_embeddings_pca.npy")
comb_image_path = os.path.join(root, "data", "embeddings", "combined_image_embeddings.npy")
pca_model_path = os.path.join(root, "data", "embeddings", "pca_model.pkl")

#Thresholds for filtering results
Text_Threshold = 0.50
Image_Threshold = 0.24

#Load compressed text and image embeddings and normalize them
embeddings = np.load(embed_path).astype("float32")
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

combined_text = np.load(comb_text_pca_path).astype("float32")
combined_text /= np.linalg.norm(combined_text, axis=1, keepdims=True)

combined_image = np.load(comb_image_path).astype("float32")
combined_image /= np.linalg.norm(combined_image, axis=1, keepdims=True)

with open(comb_meta_path, "r", encoding="utf-8") as f:
    combined_meta = json.load(f)

with open(seg_path, "r", encoding="utf-8") as f:
    segments = json.load(f)

"""
Checking for consistency
assert len(embeddings) == len(segments), "Mismatch in transcript embeddings and segments!"
assert combined_text.shape[0] == len(combined_meta), "Mismatch: combined_text and metadata"
assert combined_image.shape[0] == len(combined_meta), "Mismatch: combined_image and metadata"
"""

pca = joblib.load(pca_model_path)

def embed_text_query(text, tokenizer, text_model):
    """
    Embed a text query using the SFR-Mistral model and compress it via PCA.
    Returns:
        np.ndarray: L2-normalized compressed query vector (shape: [544])
    """

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    device = next(text_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = text_model(**inputs)
        emb = output.last_hidden_state.mean(dim=1)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1).cpu().numpy()
    compressed = pca.transform(emb)
    return compressed.squeeze() / (np.linalg.norm(compressed) + 1e-9)

def embed_clip_query(text, clip_processor, clip_model):
    """
    Embed a text query using the CLIP model (text branch).
    Returns:
        np.ndarray: L2-normalized query vector (shape: [512])
    """
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    device = next(clip_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = clip_model.get_text_features(**inputs)
        return (output / output.norm(dim=-1, keepdim=True)).squeeze().cpu().numpy()


def search_text_pgvector_ivfflat(query_vec, k=10):
    """
    Perform IVFFLAT cosine search for a text query in PostgreSQL pgvector.
    """
    conn = psycopg2.connect(dbname=data_base, user=USER, password=PASSWORD, host=HOST, port=PORT)
    cur = conn.cursor()
    vec_str = "[" + ",".join(map(str, query_vec.tolist())) + "]"
    cur.execute("""SELECT id, start, text, 1 - (embedding <-> %s) / 2.0 AS similarity
        FROM transcript_segments
        ORDER BY embedding <-> %s
        LIMIT %s;""", (vec_str, vec_str, k))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [(r[0], r[1], r[2], round(r[3], 4)) for r in rows if r[3] >= Text_Threshold]

def search_image_pgvector_ivfflat(query_vec, k=10):
    """
    Perform IVFFLAT cosine search for an image query in PostgreSQL pgvector.
    """
    conn = psycopg2.connect(dbname=data_base, user=USER, password=PASSWORD, host=HOST, port=PORT)
    cur = conn.cursor()
    vec_str = "[" + ",".join(map(str, query_vec.tolist())) + "]"
    cur.execute("""SELECT id, timestamp, image_path, 1 - (embedding <-> %s) / 2.0 AS similarity
        FROM keyframes
        ORDER BY embedding <-> %s
        LIMIT %s;""", (vec_str, vec_str, k))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [(r[0], r[1], r[2], round(r[3], 4)) for r in rows if r[3] >= Image_Threshold]



def run_pgvector_ivfflat_multimodal(query, k, tokenizer, text_model, clip_processor, clip_model, alpha=0.5):
    """
    Performs fusion search using cosine similarity between the query and precomputed combined text/image embeddings.
    Args:
        query (str): user query
        k (int): top-k results to return
        alpha (float): weighting factor for text and image scores

    Returns:
        Tuple of (text_results, image_results, fused_results)
    """
    text_vec = embed_text_query(query, tokenizer, text_model)
    image_vec = embed_clip_query(query, clip_processor, clip_model)

    text_vec /= np.linalg.norm(text_vec) + 1e-9
    image_vec /= np.linalg.norm(image_vec) + 1e-9
    text_scores = np.dot(combined_text, text_vec)
    image_scores = np.dot(combined_image, image_vec)
    fused_scores = alpha * text_scores + (1 - alpha) * image_scores

    meta_len = len(combined_meta)
    top_indices = [i for i in np.argsort(fused_scores)[::-1] if i < meta_len][:k]

    text_results = [(combined_meta[i]["timestamp"], combined_meta[i]["text"], round(text_scores[i], 4)) for i in top_indices]
    image_results = [(combined_meta[i]["timestamp"], combined_meta[i]["image_path"], round(image_scores[i], 4)) for i in top_indices]
    fused = [{"timestamp": combined_meta[i]["timestamp"],
        "text": combined_meta[i]["text"],
        "image_path": combined_meta[i]["image_path"],
        "fused_score": round(fused_scores[i], 4)} for i in top_indices]

    return text_results, image_results, fused


def run_pgvector_ivfflat(query, k, tokenizer, text_model, clip_processor, clip_model, mode="Text", alpha=0.5):
    """
    Running IVFFLAT-based retrieval for a given query.
    
    Args:
        query (str): input query string
        mode (str): one of "Text", "Image", "Multimodal Fusion"
    
    Returns:
        Tuple of (text_results, image_results, fused_results)
    """
    if mode == "Text":
        text_vec = embed_text_query(query, tokenizer, text_model)
        text_vec /= np.linalg.norm(text_vec) + 1e-9
        scores = np.dot(embeddings, text_vec)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(segments[i]["start"], segments[i]["text"], round(scores[i], 4)) for i in top_indices], [], []

    elif mode == "Image":
        image_vec = embed_clip_query(query, clip_processor, clip_model)
        image_vec /= np.linalg.norm(image_vec) + 1e-9
        scores = np.dot(combined_image, image_vec)
        top_indices = np.argsort(scores)[::-1][:k]
        return [], [(combined_meta[i]["timestamp"], combined_meta[i]["image_path"], round(scores[i], 4)) for i in top_indices], []

    elif mode == "Multimodal Fusion":
        return run_pgvector_ivfflat_multimodal(query, k, tokenizer, text_model, clip_processor, clip_model, alpha)

    return [], [], []

print("IVFFLAT fusion retrieval done")

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
                print("Invalid alpha. Defaulting to 0.5.")

        try:
            k = int(input("How many top results? (k): ").strip())
        except ValueError:
            print("Invalid value for k. Defaulting to 5.")
            k = 5

        transcript_results, image_results, fused_matches = run_pgvector_ivfflat(
            query, k, tokenizer, text_model, clip_processor, clip_model, mode=mode.title(), alpha=alpha
        )

        print("\n--- Top Results ---")
        if mode == "text":
            if transcript_results and transcript_results[0][2] >= Text_Threshold:
                for start, text, score in transcript_results:
                    print(f"[Text] ({start:.2f}s): {text} (score: {score:.4f})")
            else:
                print("No relevant text results (below threshold).\n")

        elif mode == "image":
            if image_results and image_results[0][2] >= Image_Threshold:
                for ts, path, score in image_results:
                    print(f"[Image] ({ts:.2f}s): {path} (score: {score:.4f})")
            else:
                print("No relevant image results (below threshold).\n")

        elif mode == "multimodal fusion":
            if fused_matches:
                for item in fused_matches:
                    print(f"[Fusion] ({item['timestamp']:.2f}s): {item['text']} -> {item['image_path']} (score: {item['fused_score']:.4f})")
            else:
                print("No fused matches found.\n")

        print("-" * 40 + "\n")


if __name__ == "__main__":
    main()
"""
