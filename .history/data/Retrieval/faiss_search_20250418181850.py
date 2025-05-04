import os
import faiss
import numpy as np
import re
import json
import torch
from transformers import AutoTokenizer, AutoModel

# ----------------------------
# CONFIG
# ----------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EMBED_DIR = os.path.join(ROOT_DIR, "data", "embeddings")
TEXT_EMB_PATH = os.path.join(EMBED_DIR, "text_embeddings.npy")
IMAGE_EMB_PATH = os.path.join(EMBED_DIR, "image_embeddings.npy")
SEGMENT_FILE = os.path.join(ROOT_DIR, "data", "transcripts", "segments.json")
KEYFRAME_META = os.path.join(ROOT_DIR, "data", "keyframes", "metadata.json")

TEXT_MODEL_NAME = "intfloat/e5-large-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Load data
# ----------------------------
text_emb = np.load(TEXT_EMB_PATH).astype("float32")
img_emb = np.load(IMAGE_EMB_PATH).astype("float32")

with open(SEGMENT_FILE, "r", encoding="utf-8") as f:
    segments = json.load(f)

with open(KEYFRAME_META, "r", encoding="utf-8") as f:
    keyframes = json.load(f)

# ----------------------------
# Build FAISS Indexes
# ----------------------------
def build_index(embeddings, use_cosine=True):
    if use_cosine:
        faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

text_index = build_index(text_emb)
image_index = build_index(img_emb)

# ----------------------------
# Embedding Function for Query
# ----------------------------
def embed_query(text, model, tokenizer):
    prompt = f"query: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        output = model(**inputs)
        emb = output.last_hidden_state.mean(dim=1)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.squeeze().cpu().numpy().astype("float32")

# ----------------------------
# Search Functions
# ----------------------------
def search_transcripts(query_vec, k=5):
    query_vec = query_vec.reshape(1, -1)
    faiss.normalize_L2(query_vec)
    D, I = text_index.search(query_vec, k)
    return [(segments[i], float(D[0][j])) for j, i in enumerate(I[0])]


def search_keyframes(query_vec, k=3):
    faiss.normalize_L2(query_vec)
    D, I = image_index.search(query_vec.reshape(1, -1), k)
    return [(keyframes[i], float(D[0][j])) for j, i in enumerate(I[0])]

# ----------------------------
# Main
# ----------------------------
def extract_timestamp(query):
    # Match patterns like "2:00", "12:34", or "minute 5"
    time_match = re.search(r'(\d+):(\d+)', query)
    if time_match:
        minutes = int(time_match.group(1))
        seconds = int(time_match.group(2))
        return minutes * 60 + seconds
    minute_match = re.search(r'minute\s*(\d+)', query)
    if minute_match:
        return int(minute_match.group(1)) * 60
    return None

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    model = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(DEVICE).eval()

    print("Enter your question about the video:")
    user_query = input("> ")

    timestamp = extract_timestamp(user_query)

    if timestamp is not None:
        # Timestamp-based retrieval
        closest_seg = min(segments, key=lambda seg: abs(seg["start"] - timestamp))
        print(f"\nClosest segment to {timestamp:.0f}s:")
        print(f"[{closest_seg['start']:.2f}s] {closest_seg['text']}")
    else:
        # Semantic retrieval
        query_vec = embed_query(user_query, model, tokenizer)
        results = search_transcripts(query_vec)

        print("\nTop matching transcript segments:\n")
        for seg, score in results:
            print(f"[{seg['start']:.2f}s] {seg['text']}  [score={score:.4f}]")

