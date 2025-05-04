import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
import clip
from huggingface_hub import login

# Optional: you can replace this with a secure read from an environment variable
HF_TOKEN = "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"
login(token=HF_TOKEN)


# ----------------------------
# CONFIGURATION
# ----------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEGMENT_FILE = os.path.join(ROOT_DIR, "data", "transcripts", "segments.json")
KEYFRAME_META = os.path.join(ROOT_DIR, "data", "keyframes", "metadata.json")
KEYFRAME_DIR = os.path.join(ROOT_DIR, "data", "keyframes", "images")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "embeddings")

TEXT_MODEL_NAME = "nvidia
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# TEXT EMBEDDING: E5-LARGE-V2
# ----------------------------
def embed_text_chunks(segments):
    print(" Embedding transcript segments using nv...")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    model = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(DEVICE)

    embeddings = []
    cleaned_segments = []

    for seg in tqdm(segments):
        text = seg["text"].strip()
        prompt_text = f"passage: {text}"
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)

        with torch.no_grad():
            output = model(**inputs)
            emb = output.last_hidden_state.mean(dim=1)  # mean pooling
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            embeddings.append(emb.squeeze().cpu().numpy())
            cleaned_segments.append(seg)

    np.save(os.path.join(OUTPUT_DIR, "text_embeddings.npy"), np.array(embeddings))
    print(f" Saved {len(embeddings)} text embeddings.")
    return cleaned_segments, embeddings

# ----------------------------
# IMAGE EMBEDDING: CLIP
# ----------------------------
def embed_keyframes(metadata):
    print("Embedding keyframes using CLIP...")
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)

    embeddings = []
    cleaned_frames = []

    for frame in tqdm(metadata):
        image_path = frame["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embeddings.append(image_features.squeeze().cpu().numpy())
                cleaned_frames.append(frame)
        except Exception as e:
            print(f" Skipped {image_path}: {e}")

    np.save(os.path.join(OUTPUT_DIR, "image_embeddings.npy"), np.array(embeddings))
    print(f" Saved {len(embeddings)} image embeddings.")
    return cleaned_frames, embeddings

# ----------------------------
# ALIGN: Match Each Frame to Transcript Segment
# ----------------------------
def align_frames_to_text(frames, segments):
    print(" Aligning keyframes to nearest transcript segments...")
    alignment = []

    for frame in frames:
        frame_time = frame["timestamp"]
        closest_seg = min(segments, key=lambda s: abs(s["start"] - frame_time))

        alignment.append({
            "image_path": frame["image_path"],
            "timestamp": frame_time,
            "text": closest_seg["text"],
            "text_start": closest_seg["start"],
            "text_end": closest_seg["end"]
        })

    with open(os.path.join(OUTPUT_DIR, "combined_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(alignment, f, indent=2)

    print(f" Aligned {len(alignment)} frame-text pairs.")
    return alignment

# ----------------------------
# MAIN PIPELINE
# ----------------------------
if __name__ == "__main__":
    print(" Starting embedding pipeline...")

    with open(SEGMENT_FILE, "r", encoding="utf-8") as f:
        segments = json.load(f)

    with open(KEYFRAME_META, "r", encoding="utf-8") as f:
        keyframes = json.load(f)

    cleaned_segments, _ = embed_text_chunks(segments)
    cleaned_frames, _ = embed_keyframes(keyframes)
    align_frames_to_text(cleaned_frames, cleaned_segments)

    print(" All embeddings and metadata saved successfully.")
