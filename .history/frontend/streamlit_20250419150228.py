import streamlit as st

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


