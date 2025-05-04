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
# Ask user for a query
# ----------------------------
print("Ask your question about the video:")
query = input("> ")

# TF-IDF retrieval
q_vec = tfidf_vectorizer.transform([query])
tfidf_scores = (tfidf_matrix @ q_vec.T).toarray().flatten()
top_tfidf = np.argsort(tfidf_scores)[::-1][:5]

# BM25 retrieval
bm25_scores = bm25.get_scores(query.split())
top_bm25 = np.argsort(bm25_scores)[::-1][:5]

# ----------------------------
# Display results
# ----------------------------
print("\n--- TF-IDF Top Matches ---")
for idx in top_tfidf:
    print(f"[{timestamps[idx]:.2f}s] {docs[idx]}  [score={tfidf_scores[idx]:.4f}]")

print("\n--- BM25 Top Matches ---")
for idx in top_bm25:
    print(f"[{timestamps[idx]:.2f}s] {docs[idx]}  [score={bm25_scores[idx]:.4f}]")



