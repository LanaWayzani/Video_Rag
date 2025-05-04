import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"  #Disabling file watcher for better performance in Streamlit
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"  #Reduce PyTorch memory footprint
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import re
from nltk.corpus import stopwords
from datetime import datetime
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
from data.Retrieval.faiss_search import run_retrieval as run_faiss
from data.Retrieval.pgvector_fusion_ivf import run_pgvector_ivfflat
from data.Retrieval.pgvector_fusion_hnsw import run_pgvector_hnsw
from data.Retrieval.lexical import run_lexical


st.set_page_config(page_title="Video RAG System", layout="wide")
stp_words = set(stopwords.words("english"))

content_words = 2
def low_info_query(query, min_words=content_words):
    tokens = re.findall(r"\b\w+\b", query.lower())
    content_words = [word for word in tokens if word not in stp_words]
    return len(content_words) < min_words

Text_model = "Salesforce/SFR-Embedding-Mistral"
clip_model_image = "openai/clip-vit-base-patch32"
Device = "cpu"  

@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(Text_model)
    text_model = AutoModel.from_pretrained(Text_model).to(Device)
    clip_model = CLIPModel.from_pretrained(clip_model_image).to(Device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_image)
    return tokenizer, text_model, clip_model, clip_processor

tokenizer, text_model, clip_model, clip_processor = load_models()


def keyword_in_transcript(query, transcript_text, min_overlap=2):
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    transcript_words = set(re.findall(r'\b\w+\b', transcript_text.lower()))
    return len(query_words & transcript_words) >= min_overlap

def highlight_matched_keywords(text, query):
    query_words = set(word for word in re.findall(r'\b\w+\b', query.lower()) if word not in stp_words)
    return " ".join([f"<mark style='background-color: #fff3cd'>{word}</mark>" if word.lower() in query_words else word
        for word in re.findall(r'\b\w+\b', text)])



#Setting up Streamlit app configuration
st.title(" Video RAG Retrieval System")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


st.sidebar.info(
    "**Query Modes:**\n"
    "- **Text**: Retrieves transcript segments semantically similar to the query.\n"
    "- **Image**: Retrieves keyframes by encoding the text query into visual space (CLIP).\n"
    "- **Multimodal Fusion**: Scores both transcript and image similarity, then fuses them.\n\n"
    "**Retrieval Backends:**\n"
    "- **FAISS** (Facebook AI Similarity Search): Performs fast approximate nearest neighbor search using in-memory vector indices with cosine similarity.\n"
    "- **pgvector IVFFLAT**: Uses PostgreSQL with the pgvector extension and **Inverted File Index** (IVF), partitions the vector space into clusters and searches within top candidates.\n"
    "- **pgvector HNSW**: Uses PostgreSQL with **Hierarchical Navigable Small World graphs**, a graph-based ANN method that balances accuracy and speed for large vector sets.\n"
    "- **Lexical (TF-IDF / BM25)**: Traditional keyword-based retrieval using term frequency-inverse document frequency or BM25 scoring.\n\n"
    "Lexical backend only supports **text queries**, and does not use embeddings."
)


backend = st.sidebar.selectbox("Select Retrieval Backend:", ["FAISS", "pgvector IVFFLAT", "pgvector HNSW", "Lexical (TF-IDF / BM25)"])

if backend == "Lexical (TF-IDF / BM25)":
    method = st.sidebar.radio("Lexical Method:", ["BM25", "TFIDF"]).lower()
    query_type = "Text"
    st.sidebar.markdown("Lexical search only supports **text-based** queries.")
else:
    query_type = st.sidebar.selectbox("Choose Query Type:", ["Text", "Image", "Multimodal Fusion"])
    method = None


top_k = st.sidebar.slider("Top-K Results:", min_value=1, max_value=10, value=5)
st.sidebar.markdown("### Optional Settings")

#Fusion alpha weight if the query type is Multimodal Fusion
alpha = st.sidebar.slider("Fusion Weight (alpha)", 0.0, 1.0, 0.5, step=0.1) if query_type == "Multimodal Fusion" else 0.5
text_thresh = st.sidebar.slider("Text Similarity Threshold", 0.0, 1.0, 0.25, step=0.01)
image_thresh = st.sidebar.slider("Image Similarity Threshold", 0.0, 1.0, 0.25, step=0.01) if query_type != "Text" and backend != "Lexical (TF-IDF / BM25)" else 0.25

#User query input excluding stopwords
with st.form(key="query_form"):
    query_input = st.text_input("Enter your text query:", "")
    submit = st.form_submit_button("Submit")
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if query_input and low_info_query(query_input):
    st.warning("The query is too vague. Please rephrase with more specific terms.")
    with open(os.path.join(root, "rejected_queries.log"), "a", encoding="utf-8") as log_file:
        log_file.write(f"[{datetime.now()}] REJECTED: {query_input.strip()}\n")

    st.stop()

if submit and query_input:
    st.markdown("### Current Settings")
    st.write(f"**Modality:** {query_type}")
    st.write(f"**Backend:** {backend}")
    st.write(f"**Query:** {query_input}")
    st.write(f"**Top-K:** {top_k}")

    transcript_results, image_results, fused_matches = [], [], []

    if backend == "FAISS":
        transcript_results, image_results, fused_matches = run_faiss(query_input, top_k, tokenizer, text_model, clip_processor, clip_model, mode=query_type, alpha=alpha)

    elif backend == "pgvector IVFFLAT":
        transcript_results, image_results, fused_matches = run_pgvector_ivfflat(query_input, top_k, tokenizer, text_model, clip_processor, clip_model, mode=query_type, alpha=alpha)

    elif backend == "pgvector HNSW":
        transcript_results, image_results, fused_matches = run_pgvector_hnsw(query_input, top_k, tokenizer, text_model, clip_processor, clip_model, mode=query_type, alpha=alpha)
    
    elif backend == "Lexical (TF-IDF / BM25)":
        st.write(f"**Lexical Method:** {method.upper()}")
        transcript_results = run_lexical(query_input, top_k, method=method)
    
    #Filtering results based on thresholds 
    def keyword_overlap(text, query, min_overlap=2):
        query_words = set(re.findall(r'\b\w+\b', query.lower())) - stp_words
        text_words = set(re.findall(r'\b\w+\b', text.lower()))
        return len(query_words & text_words) >= min_overlap

    if query_type == "Multimodal Fusion":
        fusion_thresh = alpha * image_thresh + (1 - alpha) * text_thresh
        fused_matches = [m for m in fused_matches if m.get("fused_score", 0) >= fusion_thresh]

        if not fused_matches:
            st.warning("No relevant matches found after applying fusion threshold. Kindly try rephrasing your query.")
            st.stop()

        top_match = fused_matches[0]
        top_score = top_match.get("fused_score", 0)
        top_text = top_match.get("text", "")

        keyword_ok = keyword_overlap(top_text, query_input, min_overlap=2)
        short_text = len(re.findall(r'\b\w+\b', top_text)) < 5
        weak_score = top_score < (fusion_thresh + 0.1)  

        if weak_score or short_text or not keyword_ok:
            st.warning("Top fused result is too weak, short, or lacks relevance. Please rephrase with more specific terms.")
            with open(os.path.join(root, "rejected_queries.log"), "a", encoding="utf-8") as log_file:
                log_file.write(f"[{datetime.now()}] FUSED-REJECT: {query_input.strip()} — Score: {top_score:.4f}, Text: {top_text[:40]}\n")
            st.stop()

    is_text_valid = transcript_results and transcript_results[0][2] >= text_thresh and keyword_in_transcript(query_input, transcript_results[0][1])
    is_image_valid = image_results and image_results[0][2] >= image_thresh

    if (query_type == "Text" and not is_text_valid) or \
       (query_type == "Image" and not is_image_valid) or \
       (query_type == "Multimodal Fusion" and not fused_matches):
        st.warning("No relevant matches found. Try rephrasing your query.")
        st.stop()

    if query_type == "Multimodal Fusion" and fused_matches:
        st.markdown("##  Fused Matches (Text with Image)")
        for match in fused_matches[:top_k]:
            show_score = match.get("fused_score", 0)
            show_only_fused = backend in {"pgvector IVFFLAT", "pgvector HNSW"}

            with st.expander(f"Fused Score: {show_score:.4f}"):
                if not show_only_fused:
                    st.markdown(f"**Transcript at {match['timestamp']:.2f}s:**")

                highlighted = highlight_matched_keywords(match['text'], query_input)
                if "<mark" not in highlighted:
                    st.caption("This fused match is based on semantic similarity. No exact keywords matched.")
                st.markdown(highlighted, unsafe_allow_html=True)

                ts = match['timestamp'] if match['timestamp'] > 0.0 else 3.0
                img_path = match["image_path"]
                if match["timestamp"] == 0.0:
                    img_path = re.sub(r"frame_0\.0s\.(jpg|png)", r"frame_3.0s.\1", img_path)
                img_path = os.path.join(root, img_path)
                if os.path.exists(img_path):
                    st.image(img_path, caption=f"{ts:.2f}s | Keyframe", use_container_width=True)

    if transcript_results and not (backend in {"pgvector IVFFLAT", "pgvector HNSW"} and query_type == "Multimodal Fusion"):
        st.markdown("##  Transcript Matches")
        for start, text, score in transcript_results:
           with st.expander(f"Text Score: {score:.4f}"):
                st.markdown(f"**[{start:.2f}s]**", unsafe_allow_html=True)
                highlighted = highlight_matched_keywords(text, query_input)
                if "<mark" not in highlighted and backend in {"FAISS", "pgvector IVFFLAT", "pgvector HNSW"}:
                    st.caption("This result is shown based on semantic similarity, no exact keyword matches.")
            
                st.markdown(highlighted, unsafe_allow_html=True)


    if image_results and (query_type == "Image" or (query_type == "Multimodal Fusion" and backend not in {"pgvector IVFFLAT", "pgvector HNSW"})):
        st.markdown("##  Keyframe Matches")
        for ts_raw, img_path, score in image_results:
            ts = ts_raw if ts_raw > 0.0 else 3.0
            with st.expander(f"Image Score: {score:.4f}"):
                full_path = os.path.join(root, img_path)
                if os.path.exists(full_path):
                    st.image(full_path, caption=f"Keyframe at {ts:.2f}s", use_container_width=True)
                else:
                    st.warning(f"Image not found: {img_path}")
    
    st.session_state.chat_history.append({"query": query_input,"backend": backend,"type": query_type,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"transcripts": transcript_results,"images": image_results,
    "fused": fused_matches})

    # Video playback 
    video_path = os.path.join(os.path.dirname(__file__), "static", "complexity.mp4")
    if transcript_results:
        best_time = transcript_results[0][0]
    elif fused_matches and query_type == "Multimodal Fusion":
        best_time = fused_matches[0]["timestamp"]
    elif image_results and query_type == "Image":
        best_time = image_results[0][0]
    else:
        best_time = 0.0
    if best_time == 0.0:
        best_time = 3.0
    formatted_time = f"{int(best_time // 60)}:{int(best_time % 60):02d}"

    if os.path.exists(video_path):
        if query_type == "Text":
            st.caption(" Time reference is based on the top transcript match.")
        elif query_type == "Image":
            st.caption(" Time reference is based on the top keyframe match.")
        elif query_type == "Multimodal Fusion":
            st.caption("Time reference is based on the top fused match.")
        
        st.markdown(f"Kindly refer to **{formatted_time}** in the video below:")
        with open(video_path, "rb") as f:
            st.video(f.read(), format="video/mp4")
    else:
        st.warning("Video file not found.")
 

if st.button(" Clear Query History"):
    st.session_state.chat_history.clear()
    st.rerun()

if st.session_state.chat_history:
    st.markdown("## Previous Queries")
    for i, entry in enumerate(reversed(st.session_state.chat_history), 1):
        with st.expander(f"{i}. {entry['query']} — {entry['backend']} | {entry['type']}"):
            st.markdown(f"**Timestamp:** {entry['timestamp']}")
            if entry["transcripts"]:
                st.markdown("**Transcript Matches:**")
                for ts, text, score in entry["transcripts"]:
                    st.markdown(f"• [{ts:.2f}s] {text} (score: {score:.2f})")
            if entry["images"]:
                st.markdown("**Image Matches:**")
                for ts, path, score in entry["images"]:
                    st.markdown(f"• [{ts:.2f}s] {path} (score: {score:.2f})")
            if entry["fused"]:
                st.markdown("**Fused Matches:**")
                for match in entry["fused"]:
                    st.markdown(f"• [{match['timestamp']:.2f}s] {match['text']} -> {match['image_path']} (score: {match['fused_score']:.2f})")


