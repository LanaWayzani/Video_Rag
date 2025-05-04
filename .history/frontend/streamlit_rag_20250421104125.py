import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Retrieval functions
from data.Retrieval.faiss_search import run_retrieval
from data.Retrieval.pgvector_fusion_ivf import run_pgvector_ivfflat_fusion
from data.Retrieval.pgvector_fusion_hnsw import run_pgvector_hnsw_fusion
from data.Retrieval.lexical import run_lexical

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="Video RAG System", layout="wide")
st.title("🎬 Video RAG Retrieval System")

# ----------------------------
# Sidebar Options
# ----------------------------
st.sidebar.header("🔍 Retrieval Choices")

query_type = st.sidebar.selectbox("Choose query type:", ["Text", "Multimodal"])
backend = st.sidebar.selectbox("Select retrieval method:", [
    "FAISS", "pgvector IVFFLAT", "pgvector HNSW", "Lexical (TF-IDF / BM25)"
])
top_k = st.sidebar.slider("Number of results (Top-K):", min_value=1, max_value=10, value=5)
query_input = st.text_input("What do you want to ask about the video?:", "")

# ----------------------------
# Path Constants
# ----------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ----------------------------
# Retrieval Logic
# ----------------------------
if query_input:
    st.markdown("### 🔄 Current Query Settings")
    st.write(f"**Modality**: {query_type}")
    st.write(f"**Backend**: {backend}")
    st.write(f"**Query**: {query_input}")
    st.write(f"**Top-K**: {top_k}")

    # Run retrieval
    if backend == "FAISS":
        transcript_results, image_results = run_retrieval(query_input, top_k)
    elif backend == "pgvector IVFFLAT":
        transcript_results, image_results = run_pgvector_ivfflat_fusion(query_input, top_k)
    elif backend == "pgvector HNSW":
        transcript_results, image_results = run_pgvector_hnsw_fusion(query_input, top_k)
    elif backend == "Lexical (TF-IDF / BM25)":
        transcript_results, _ = run_lexical(query_input, top_k)

    # Check for no matches
    if not transcript_results or (query_type == "Multimodal" and backend != "Lexical (TF-IDF / BM25)" and not image_results):
        st.warning("⚠️ No relevant match found. Try rephrasing your question.")
        st.stop()

    # Show transcript results
    st.markdown("### 📄 Transcript Matches")
    for start, text, score in transcript_results:
        st.markdown(f"**[{start:.2f}s]** {text}  `score={score:.4f}`")
    st.markdown("Video Preview")
    video_path = os.path.join(ROOT, "data", "raw_video", "complexity.mp4")
    if os.path.exists(video_path):
        st.video(video_path)
        best_time = transcript_results[0][0]
        formatted_time = f"{int(best_time // 60)}:{int(best_time % 60):02d}"
        st.markdown(f"⏱️ You can **seek to {formatted_time}** in the video for the best match.")
    else:
        st.warning("Video file not found.")

    # Show image results
    if query_type == "Multimodal" and backend != "Lexical (TF-IDF / BM25)":
        st.markdown("Image Matches")
        for ts, img_path, score in image_results:
            full_path = os.path.join(ROOT, img_path)
            if os.path.exists(full_path):
                st.image(full_path, caption=f"{ts:.2f}s | score={score:.4f}", use_container_width=True)
            else:
                st.warning(f"Image not found at: {img_path}")

