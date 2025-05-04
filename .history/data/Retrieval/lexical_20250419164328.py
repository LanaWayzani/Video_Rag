import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

# ----------------------------
# CONFIG
# ----------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEGMENT_PATH = os.path.join(ROOT, "data", "transcripts", "segments.json")

# ----------------------------
# Load transcript segments
# ----------------------------
with open(SEGMENT_PATH, "r", encoding="utf-8") as f:
    segments = json.load(f)

docs = [seg["text"] for seg in segments]
timestamps = [seg["start"] for seg in segments]

# ----------------------------
# TF-IDF setup
# ----------------------------
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(docs)

# ----------------------------
# BM25 setup
# ----------------------------
tokenized_docs = [doc.split() for doc in docs]
bm25 = BM25Okapi(tokenized_docs)

# ----------------------------
# Lexical search function
# ----------------------------
def run_lexical(query, k=5):
    tfidf_vec = tfidf_vectorizer.transform([query])
    tfidf_scores = (tfidf_matrix @ tfidf_vec.T).toarray().flatten()
    tfidf_top_idx = np.argsort(tfidf_scores)[::-1][:k]

    bm25_scores = bm25.get_scores(query.split())
    bm25_top_idx = np.argsort(bm25_scores)[::-1][:k]

    tfidf_results = [(timestamps[i], docs[i], tfidf_scores[i]) for i in tfidf_top_idx]
    bm25_results = [(timestamps[i], docs[i], bm25_scores[i]) for i in bm25_top_idx]

    return tfidf_results, bm25_results

