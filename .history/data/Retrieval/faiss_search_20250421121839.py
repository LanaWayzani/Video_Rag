import os
import faiss
import numpy as np
import re
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel

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
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIMILARITY_THRESHOLD = 0.3  # can adjust this later

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
# Embedding Functions
# ----------------------------
def embed_query(text, model, tokenizer):
    prompt = f"query: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        output = model(**inputs)
        emb = output.last_hidden_state.mean(dim=1)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.squeeze().cpu().numpy().astype("float32")

def embed_clip_query(text, processor, model):
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        output = model.get_text_features(**inputs)
        output = torch.nn.functional.normalize(output, p=2, dim=1)
    return output.squeeze().cpu().numpy().astype("float32")

# ----------------------------
# Search Functions
# ----------------------------
def search_transcripts(query_vec, k=5):
    query_vec = query_vec.reshape(1, -1)
    faiss.normalize_L2(query_vec)
    D, I = text_index.search(query_vec, k)
    return [(segments[i], float(D[0][j])) for j, i in enumerate(I[0])]

def search_keyframes(query_vec, k=3):
    query_vec = query_vec.reshape(1, -1)
    faiss.normalize_L2(query_vec)
    D, I = image_index.search(query_vec, k)
    return [(keyframes[i], float(D[0][j])) for j, i in enumerate(I[0])]

# ----------------------------
# Timestamp Extraction
# ----------------------------
def extract_timestamp(query):
    time_match = re.search(r'(\d+):(\d+)', query)
    if time_match:
        minutes = int(time_match.group(1))
        seconds = int(time_match.group(2))
        return minutes * 60 + seconds
    minute_match = re.search(r'minute\s*(\d+)', query)
    if minute_match:
        return int(minute_match.group(1)) * 60
    return None

# ----------------------------
# Main CLI Entry
# ----------------------------
if __name__ == "__main__":
    # Load both models
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    model = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(DEVICE).eval()

    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE).eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    print("Enter your question about the video:")
    user_query = input("> ")

    timestamp = extract_timestamp(user_query)

    if timestamp is not None:
        closest_seg = min(segments, key=lambda seg: abs(seg["start"] - timestamp))
        print(f"\nClosest segment to {timestamp:.0f}s:")
        print(f"[{closest_seg['start']:.2f}s] {closest_seg['text']}")
    else:
        # Transcript search
        query_vec = embed_query(user_query, model, tokenizer)
        text_results = search_transcripts(query_vec)
        if text_results[0][1] < SIMILARITY_THRESHOLD:
            print("\n No confident transcript match found.")
        else:
            print("\n--- Transcript Matches ---")
            for seg, score in text_results:
                print(f"[{seg['start']:.2f}s] {seg['text']}  [score={score:.4f}]")

        # Keyframe search
        clip_query_vec = embed_clip_query(user_query, clip_processor, clip_model)
        image_results = search_keyframes(clip_query_vec)
        if image_results[0][1] < SIMILARITY_THRESHOLD:
            print("\n No confident keyframe match found.")
        else:
            print("\n--- Keyframe Matches ---")
            for i, (frame, score) in enumerate(image_results):
                time_val = frame.get("time") or frame.get("timestamp") or -1
                print(f"[{time_val:.2f}s] {frame['image_path']}  [score={score:.4f}]")
                # Preview only top 1–2 images
                if i < 2:
                    image_path = os.path.join(ROOT_DIR, frame["image_path"])
                    if os.path.exists(image_path):
                        img = Image.open(image_path)
                        plt.figure()
                        plt.imshow(img)
                        plt.axis("off")
                        plt.title(f"{time_val:.1f}s  |  score={score:.4f}")
                        plt.show()
                    else:
                        print(f" Image not found at: {image_path}")
SIMILARITY_THRESHOLD = 0.3  

def run_retrieval(query, k=5):
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    model = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(DEVICE).eval()

    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE).eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    text_vec = embed_query(query, model, tokenizer)
    image_vec = embed_clip_query(query, clip_processor, clip_model)

    text_matches = search_transcripts(text_vec, k)
    image_matches = search_keyframes(image_vec, k)

    best_text_score = text_matches[0][1] if text_matches else 0.0
    best_image_score = image_matches[0][1] if image_matches else 0.0

    if best_text_score < SIMILARITY_THRESHOLD:
        print(f"🔴 Rejected transcript (score={best_text_score:.4f}) — below threshold")
        transcript_results = []
    else:
        transcript_results = [(seg["start"], seg["text"], score) for seg, score in text_matches]

    if best_image_score < SIMILARITY_THRESHOLD:
        print(f"🔴 Rejected keyframe (score={best_image_score:.4f}) — below threshold")
        image_results = []
    else:
        image_results = [(frame.get("time", frame.get("timestamp", -1)), frame["image_path"], score) for frame, score in image_matches]

    return transcript_results, image_results


