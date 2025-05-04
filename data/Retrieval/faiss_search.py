import os
import faiss
import numpy as np
import json
import torch
import joblib

"""
This script sets up FAISS indexes for fast similarity search over both text and image embeddings,
and supports text, image, or fused multimodal retrieval modes. 
"""

# Define paths to embedding and metadata files
Root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
embed_dir = os.path.join(Root, "data", "embeddings")
Text_embed_path = os.path.join(embed_dir, "compressed_text_embeddings.npy")  
image_embed_path = os.path.join(embed_dir, "image_embeddings.npy")
seg_file = os.path.join(Root, "data", "transcripts", "segments.json")
Keyframe_meta = os.path.join(Root, "data", "keyframes", "metadata.json")
comb_tex = os.path.join(embed_dir, "combined_text_embeddings_pca.npy")
comb_image = os.path.join(embed_dir, "combined_image_embeddings.npy")
combined_meta = os.path.join(embed_dir, "combined_metadata_aligned.json")

combined_text_emb = np.load(comb_tex).astype("float32")
combined_image_emb = np.load(comb_image).astype("float32")
with open(combined_meta, "r", encoding="utf-8") as f:
    combined_meta = json.load(f)


text_emb = np.load(Text_embed_path).astype("float32")
img_emb = np.load(image_embed_path).astype("float32")
with open(seg_file, "r", encoding="utf-8") as f:
    segments = json.load(f)
with open(Keyframe_meta, "r", encoding="utf-8") as f:
    keyframes = json.load(f)



pca = joblib.load(os.path.join(embed_dir, "pca_model.pkl"))

def build_index(embeddings, use_cosine=True):
    """
    Build a FAISS index from given embeddings.
    Args:
        embeddings (np.ndarray): Embedding vectors to index
        use_cosine (bool): If True, normalize vectors for cosine similarity

    Returns:
        faiss.IndexFlatIP: Inner product index (acts as cosine if normalized)
    """
    if use_cosine:
        faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1]) 
    index.add(embeddings)
    return index

text_index = build_index(text_emb)
image_index = build_index(img_emb)
combined_text_index = build_index(combined_text_emb)
combined_image_index = build_index(combined_image_emb)


def embed_text_query(text, tokenizer, text_model):
    """
    Convert a text query into an embedding using the given tokenizer and model.

    Args:
        text (str): Input text
        tokenizer: HuggingFace tokenizer
        text_model: HuggingFace transformer model

    Returns:
        np.ndarray: Normalized text embedding vector
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    device = text_model.device if hasattr(text_model, "device") else next(text_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = text_model(**inputs)
        emb = output.last_hidden_state.mean(dim=1)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)

    return emb.squeeze().cpu().numpy().astype("float32")

def embed_clip_query(text, clip_processor, clip_model):
    """
    Convert a text query into a CLIP-compatible text embedding.
    Args:
        text (str): Input text
        clip_processor: CLIP tokenizer/preprocessor
        clip_model: CLIP model

    Returns:
        np.ndarray: Normalized CLIP embedding vector
    """
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    device = clip_model.device if hasattr(clip_model, "device") else next(clip_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = clip_model.get_text_features(**inputs)
        output = torch.nn.functional.normalize(output, p=2, dim=1)

    return output.squeeze().cpu().numpy().astype("float32")




def search_faiss(index, query_vec, data, top_k):
    """
    Perform top-k FAISS search with cosine similarity.

    Args:
        index (faiss.Index): FAISS index to search
        query_vec (np.ndarray): Query vector
        data (list): Metadata to return for each match
        top_k (int): Number of top results to return

    Returns:
        List[Tuple[data, score]]: Ranked results
    """
    query_vec = np.ascontiguousarray(query_vec.astype("float32")).reshape(1, -1)
    faiss.normalize_L2(query_vec)  # Ensure cosine similarity
    D, I = index.search(query_vec, top_k)
    return [(data[i], float(D[0][j])) for j, i in enumerate(I[0])]

def search_transcripts(query_vec, k=5):
    return search_faiss(text_index, query_vec, segments, k)

def search_keyframes(query_vec, k=3):
    return search_faiss(image_index, query_vec, keyframes, k)




def run_multimodal_fusion(query, k, tokenizer, text_model, clip_processor, clip_model, alpha=0.5):
    """
    Performs fusion between text and image modalities.

    Args:
        query (str): User query
        k (int): Number of top results
        tokenizer, text_model: For text embedding
        clip_processor, clip_model: For CLIP embedding
        alpha (float): Fusion weight (text importance)

    Returns:
        List[Tuple[metadata, fused_score]]
    """
    text_vec = embed_text_query(query, tokenizer, text_model)
    text_vec = pca.transform(text_vec.reshape(1, -1)).squeeze().astype("float32")
    image_vec = embed_clip_query(query, clip_processor, clip_model)

    #Normalize query vectors
    text_vec = text_vec / (np.linalg.norm(text_vec) + 1e-9)
    image_vec = image_vec / (np.linalg.norm(image_vec) + 1e-9)

    #Cosine similarity through dot product since vectors are normalized
    text_scores = np.dot(combined_text_emb, text_vec)
    image_scores = np.dot(combined_image_emb, image_vec)

    #Weighted sum for fusion
    fused_scores = alpha * text_scores + (1 - alpha) * image_scores
    top_indices = np.argsort(fused_scores)[::-1][:k]

    return [(combined_meta[i], fused_scores[i]) for i in top_indices]

def run_retrieval(query, k, tokenizer, text_model, clip_processor, clip_model, mode="Text", alpha=0.5):
    """
    Unified Retrieval interface for text, image, or multimodal fusion.

    Args:
        query (str): User query string
        k (int): Top-k results to return
        tokenizer, text_model, clip_processor, clip_model: All models
        mode (str): "Text", "Image", or "Multimodal Fusion"
        alpha (float): Text weight in fusion

    Returns:
        Tuple[List, List, List]: Transcript results, Image results, Fused results
    """
    if not query.strip():
        return [], [], []

    if mode == "Text":
        text_vec = embed_text_query(query, tokenizer, text_model)
        text_vec = pca.transform(text_vec.reshape(1, -1)).squeeze().astype("float32")
        text_matches = search_transcripts(text_vec, k)
        transcript_results = [(seg["start"], seg["text"], score) for seg, score in text_matches]
        return transcript_results, [], []

    elif mode == "Image":
        image_vec = embed_clip_query(query, clip_processor, clip_model)
        image_matches = search_keyframes(image_vec, k)
        image_results = [(frame["timestamp"], frame["image_path"], score) for frame, score in image_matches]
        return [], image_results, []

    elif mode == "Multimodal Fusion":
        fusion_results = run_multimodal_fusion(query, k, tokenizer, text_model, clip_processor, clip_model, alpha)
        fused = [{
            "timestamp": item["timestamp"],
            "text": item["text"],
            "image_path": item["image_path"],
            "fused_score": score
        } for item, score in fusion_results]
        return [], [], fused

    return [], [], []

print("FAISS retrieval done")

"""
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModel
    from transformers import CLIPProcessor, CLIPModel

    # Load models once
    print("Loading models...")
    TEXT_MODEL_NAME = "Salesforce/SFR-Embedding-Mistral"
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    text_model = AutoModel.from_pretrained(TEXT_MODEL_NAME).eval()

    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).eval()

    Text_threshold = 0.65
    Image_threshold = 0.25

    print("\nType 'exit' to quit at any time.\n")

    while True:
        query = input("Enter a query: ").strip()
        if query.lower() == "exit":
            break

        mode = input("Select mode [Text / Image / Multimodal Fusion]: ").strip()
        if mode.lower() == "exit":
            break

        if mode == "Multimodal Fusion":
            alpha_input = input("Enter alpha (0–1 for fusion weight ").strip()
            if alpha_input.lower() == "exit":
                break
            alpha = float(alpha_input)
        else:
            alpha = 0.5  

        k_input = input("How many top results? (k): ").strip()
        if k_input.lower() == "exit":
            break
        k = int(k_input)

        # Run retrieval
        transcript_results, image_results, fused_matches = run_retrieval(
            query, k, tokenizer, text_model, clip_processor, clip_model, mode=mode, alpha=alpha)

        text_valid = (transcript_results and transcript_results[0][2] >= Text_threshold)
        image_valid = (image_results and image_results[0][2] >= Image_threshold )
        fusion_valid = ( mode == "Multimodal Fusion" and fused_matches)

        if (mode == "Text" and not text_valid) or \
           (mode == "Image" and not image_valid) or \
           (mode == "Multimodal Fusion" and not fusion_valid):
            print("No relevant results found. Try again.\n")
            continue

        print("\n--- Top Results ---")
        if transcript_results:
            for ts, text, score in transcript_results:
                print(f"[Text] ({ts:.2f}s): {text} (score: {score:.3f})")

        if image_results:
            for ts, path, score in image_results:
                print(f"[Image] ({ts:.2f}s): {path} (score: {score:.3f})")

        if fused_matches:
            for item in fused_matches:
                print(f"[Fusion] ({item['timestamp']:.2f}s): {item['text']} -> {item['image_path']} (score: {item['fused_score']:.3f})")

        print("" + "-" * 40 + "\n")
"""