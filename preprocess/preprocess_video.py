import os
import json
import whisper
import cv2
import torch
import clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

"""
This script is the first step towards applying Video RAG systems, it does the following:   
1. Transcribes the audio from a video file ( the lecture) into timestamped segments using OpenAI's Whisper.
2. Extracts visually distinct keyframes using CLIP embeddings with cosine similarity filtering.

It outputs:
1. JSON file of timestamped transcript segments.
2. Folder of distinct keyframe images.
3. Metadata JSON linking each keyframe to its timestamp.

"""

#Path for the video file and output directories
direc = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
video_filename = os.path.join(direc, "data", "raw_video", "complexity.mp4")
transcript_output = os.path.join(direc, "data", "transcripts", "segments.json")
frame_dir = os.path.join(direc, "data", "keyframes", "images")
metadata_path = os.path.join(direc, "data", "keyframes", "metadata.json")
#Time between frame checks (in seconds)
interval_s = 3   
#Whisper model size: base, tiny, small, medium, large             
model_size = "medium"         


def directories():
    """Ensure all output folders exist."""
    os.makedirs(os.path.join(direc, "data", "raw_video"), exist_ok=True)
    os.makedirs(os.path.join(direc, "data", "transcripts"), exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)


#This function transcribes the video using OpenAI's Whisper model
def whisper_transcript(video_path, output_json, model_size="base"):
    """
    Transcribes a video using OpenAI's Whisper model.

    Args:
        video_path (str): Path to the video file.
        output_json (str): Path to save transcript JSON.
        model_size (str): Whisper model variant to use, which in our case is "medium".
    Returns:
        List[dict]: Transcript segments with start, end, and text.
    """
    model = whisper.load_model(model_size)
    result = model.transcribe(video_path, verbose=True)

    segments = [{"start": seg["start"], "end": seg["end"], "text": seg["text"]}         
        for seg in result["segments"]]
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)
    print(f"Transcript segments saved to {output_json}")
    return segments


#This function extracts distinct keyframes using CLIP embeddings and cosine similarity
def image_keyframes(video_path, output_folder, metadata_path, interval_s=3, similarity_threshold=0.92, device='cpu'):
    """
    Extracts distinct keyframes using CLIP embeddings and cosine similarity.

    Args:
        video_path (str): Path to the video file.
        output_folder (str): location to save keyframe images.
        metadata_path (str): Path to save image metadata JSON.
        interval_s (int): Time between frame checks.
        similarity_threshold (float): Cosine similarity threshold to filter similar frames.
        device (str): "cpu" or "cuda"

    Returns:
        List[dict]: Metadata for saved keyframes (image_path, timestamp).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(interval_s * fps)
    frame_idx = 0

    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    prev_embedding = None
    keyframes_meta = []

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        image = preprocess(pil_image).unsqueeze(0).to(device)

        #Normalize the image tensor to avoid batch dimension issues 
        with torch.no_grad():
            emb = model.encode_image(image)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb_np = emb.cpu().numpy()

        #Calculate cosine similarity with the previous embedding in order to filter out similar frames
        #If the cosine similarity is below the threshold, save the frame as a keyframe
        if prev_embedding is None or cosine_similarity(emb_np, prev_embedding)[0][0] < similarity_threshold:
            img_filename = f"frame_{timestamp:.1f}s.png"
            img_path = os.path.join(output_folder, img_filename)
            cv2.imwrite(img_path, rgb_frame)
            keyframes_meta.append({"image_path": img_path, "timestamp": timestamp})
            prev_embedding = emb_np

        frame_idx += interval_frames

    cap.release()
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(keyframes_meta, f, indent=2)
    print(f"Saved {len(keyframes_meta)} unique keyframes to {output_folder}")
    print(f"Metadata saved to {metadata_path}")
    return keyframes_meta


if __name__ == "__main__":
    directories()
    segments = whisper_transcript(video_filename, transcript_output, model_size=model_size)
    keyframes = image_keyframes(video_filename, frame_dir, metadata_path, interval_s=interval_s)

