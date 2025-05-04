import streamlit as st
import importlib
import matplotlib.pyplot as plt
from Retrieval.faiss_search import run_retrieval
from Retrieval.lexical import run_lexical


from PIL import Image
import os


# Set up the Streamlit page
st.set_page_config(page_title="Video RAG System", layout="wide")

# Title
st.title("🎬 Video RAG Retrieval System")

# Sidebar for options
st.sidebar.header("🔍 Retrieval Options")

# Query modality
query_type = st.sidebar.selectbox(
    "Choose query modality:",
    ["Text", "Multimodal"]
)

# Backend selection
backend = st.sidebar.selectbox(
    "Select retrieval backend:",
    ["FAISS", "pgvector IVFFLAT", "pgvector HNSW", "Lexical (TF-IDF / BM25)"]
)

# Top-k results setting
top_k = st.sidebar.slider("Number of results (Top-K):", min_value=1, max_value=10, value=5)

# Text input
query_input = st.text_input("Ask a question about the video:", "")

# Placeholder to show options selected (we will replace with logic next)
if query_input:
    st.markdown("### 🔄 Current Query Settings")
    st.write(f"**Modality**: {query_type}")
    st.write(f"**Backend**: {backend}")
    st.write(f"**Query**: {query_input}")
    st.write(f"**Top-K**: {top_k}")

    # Placeholders for results
    st.markdown("### 📄 Transcript Matches")
    st.info("Transcript results will appear here based on selected method...")

    if query_type == "Multimodal":
        st.markdown("### 🖼️ Image Matches")
        st.info("Image keyframes will be retrieved here...")

# Path constants
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KEYFRAME_DIR = os.path.join(ROOT, "data", "keyframes", "images")

# Run retrieval logic
if query_input:
    results_module = None

    if backend == "FAISS":
        results_module = importlib.import_module("data.Retrieval.faiss_search")
    elif backend == "pgvector IVFFLAT":
        results_module = importlib.import_module("data.Retrieval.pgvector_fusion_ivf")
    elif backend == "pgvector HNSW":
        results_module = importlib.import_module("data.Retrieval.pgvector_fusion_hnsw")
    elif backend == "Lexical (TF-IDF / BM25)":
        results_module = importlib.import_module("data.Retrieval.lexical")

    if results_module:
        # Call retrieval function dynamically
        if backend.startswith("Lexical"):
            transcript_results, _ = results_module.run_lexical(query_input, top_k)
        else:
            transcript_results, image_results = results_module.run_retrieval(query_input, top_k)

        # Show transcripts
        st.markdown("### 📄 Transcript Matches")
        for start, text, score in transcript_results:
            st.markdown(f"**[{start:.2f}s]** {text}  `score={score:.4f}`")

        # Show keyframes if needed
        if query_type == "Multimodal" and backend != "Lexical (TF-IDF / BM25)":
            st.markdown("### 🖼️ Image Matches")
            for ts, img_path, score in image_results:
                full_path = os.path.join(ROOT, img_path)
                if os.path.exists(full_path):
                    st.image(full_path, caption=f"{ts:.2f}s | score={score:.4f}", use_column_width=True)
    


