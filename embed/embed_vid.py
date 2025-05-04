import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import clip

"""
This script performs a full embedding pipeline for video rag where it processes
 transcript segments and extracted video keyframes to generate:

1. Text Embeddings using the SFR-Embedding-Mistral model chosen based on the Retrieval metric in the MTEB leaderboard.
2. Image Embeddings using the CLIP ViT-B/32 model.
3. Multimodal Alignments by matching keyframe timestamps to transcript intervals.
4. Combined Embeddings for aligned image-text pairs. If there's a gap between repeated keyframes, I don't skip the transcript. I add
 all the text segments that refer to the same image in between the two unique keyframes .

"""

#Defining paths to data directories
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
seg_file = os.path.join(root, "data", "transcripts", "segments.json")
keyframes_meta = os.path.join(root, "data", "keyframes", "metadata.json")
frame_dir = os.path.join(root, "data", "keyframes", "images")
output_dir = os.path.join(root, "data", "embeddings")
os.makedirs(output_dir, exist_ok=True)


Device = "cpu"
Text_model = "Salesforce/SFR-Embedding-Mistral"


def embed_text_chunks(segments):
    """
    Computes normalized text embeddings for transcript segments using the SFR model.
    Args:
        segments (list): List of transcript segments with 'text'.

    Returns:
        cl_seg (list), embeddings (list of np.arrays)
    """
    tokenizer = AutoTokenizer.from_pretrained(Text_model)
    model = AutoModel.from_pretrained(Text_model).to(Device).eval()

    embeddings = []
    cl_seg = []

    texts = [seg["text"].strip() for seg in segments]
    """
    We generate embeddings for transcript segments using a pre-trained transformer. Each batch of texts is tokenized and 
    passed through the model. Then, we apply attention-masked mean pooling over token embeddings,followed by L2 normalization 
    to obtain fixed-size, unit-length sentence embeddings. 
    """
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), 16)):
            batch = texts[i:i + 16]
            encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(Device)
            output = model(**encoded)
            last_hidden = output.last_hidden_state
            mask = encoded.attention_mask.unsqueeze(-1).expand(last_hidden.size())
            summed = torch.sum(last_hidden * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = summed / counts
            normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            embeddings.extend(normalized.cpu().numpy())
            cl_seg.extend(segments[i:i + 16])

    np.save(os.path.join(output_dir, "text_embeddings.npy"), np.array(embeddings))
    print(f"Saved {len(embeddings)} text embeddings.")
    return cl_seg, embeddings



def embed_keyframes(metadata):
    """
    Computes normalized image embeddings for keyframes using CLIP ViT-B/32.

    Args:
        metadata (list): List of dicts with 'image_path' keys.
    Returns:
        cl_frames (list), embeddings (list of np.arrays)
    """
    model, preprocess = clip.load("ViT-B/32", device=Device)

    embeddings = []
    cl_frames = []

    #Iterate through the metadata and process each image and compute its embedding 
    for frame in tqdm(metadata):
        image_path = frame["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(Device)

            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embeddings.append(image_features.squeeze().cpu().numpy())
                cl_frames.append(frame)
        except Exception as e:
            print(f" Skipped {image_path}: {e}")

    np.save(os.path.join(output_dir, "image_embeddings.npy"), np.array(embeddings))
    print(f" Saved {len(embeddings)} image embeddings.")
    return cl_frames, embeddings


def align_frames_to_text(frames, segments):
    """
    Aligns keyframe timestamps to the closest transcript segments.

    Args:
        frames (list): List of keyframes with timestamps.
        segments (list): List of transcript segments with start/end times.

    Returns:
        alignment (list): Combined data for frame-text alignment.
    """
    alignment = []
    frames_sorted = sorted(frames, key=lambda x: x["timestamp"])

    for i, frame in enumerate(frames_sorted):
        t_start = frame["timestamp"]
        t_end = frames_sorted[i + 1]["timestamp"] if i + 1 < len(frames_sorted) else float('inf')

        matching_segs = [s for s in segments if t_start <= s["start"] < t_end]
        full_text = " ".join([s["text"].strip() for s in matching_segs])

        if full_text.strip():
            alignment.append({"image_path": frame["image_path"],
                "timestamp": t_start,
                "text": full_text,
                "text_start": matching_segs[0]["start"] if matching_segs else None,
                "text_end": matching_segs[-1]["end"] if matching_segs else None})

    with open(os.path.join(output_dir, "combined_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(alignment, f, indent=2)

    print(f"Aligned {len(alignment)} keyframes to  transcript intervals.")
    return alignment


def embed_combined_metadata(combined_meta):
    """
    Embeds aligned image-text pairs using CLIP and SFR models.

    Args:
        combined_meta (list): List of dicts with aligned 'text' and 'image_path'.

    Saves:
        combined_text_embeddings.npy
        combined_image_embeddings.npy
        combined_metadata_aligned.json
    """

    tokenizer = AutoTokenizer.from_pretrained(Text_model)
    text_model = AutoModel.from_pretrained(Text_model).to(Device).eval()
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=Device)

    text_embeddings = []
    image_embeddings = []
    valid_entries = []

    with torch.no_grad():
        for entry in tqdm(combined_meta):
            text = entry["text"].strip()
            image_path = entry["image_path"]

            try:
                encoded = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(Device)
                output = text_model(**encoded)
                last_hidden = output.last_hidden_state
                mask = encoded.attention_mask.unsqueeze(-1).expand(last_hidden.size())
                summed = torch.sum(last_hidden * mask, dim=1)
                counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                mean_pooled = summed / counts
                text_vec = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

                image = Image.open(image_path).convert("RGB")
                image_tensor = clip_preprocess(image).unsqueeze(0).to(Device)
                image_vec = clip_model.encode_image(image_tensor)
                image_vec = image_vec / image_vec.norm(dim=-1, keepdim=True)

                text_embeddings.append(text_vec.squeeze().cpu().numpy())
                image_embeddings.append(image_vec.squeeze().cpu().numpy())
                valid_entries.append(entry)

            except Exception as e:
                print(f"Skipped {image_path}: {e}")

    np.save(os.path.join(output_dir, "combined_text_embeddings.npy"), np.array(text_embeddings))
    np.save(os.path.join(output_dir, "combined_image_embeddings.npy"), np.array(image_embeddings))
    with open(os.path.join(output_dir, "combined_metadata_aligned.json"), "w", encoding="utf-8") as f:
        json.dump(valid_entries, f, indent=2)

    print(f"Saved {len(valid_entries)} aligned multimodal embeddings.")



if __name__ == "__main__":

    with open(seg_file, "r", encoding="utf-8") as f:
        segments = json.load(f)
    with open(keyframes_meta, "r", encoding="utf-8") as f:
        keyframes = json.load(f)

    cl_seg, _ = embed_text_chunks(segments)
    cl_frames, _ = embed_keyframes(keyframes)
    combined_meta = align_frames_to_text(cl_frames, cl_seg)
    embed_combined_metadata(combined_meta)

    print("Embeddings and metadata are performed and saved.")