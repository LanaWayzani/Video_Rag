import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd
import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
segment_path = os.path.join(ROOT, "data", "transcripts", "segments.json")

with open(segment_path, "r", encoding="utf-8") as f:
    segments = json.load(f)

# Prepare corpus
docs = [seg["text"] for seg in segments]
timestamps = [seg["start"] for seg in segments]

# --- TF-IDF Search Setup ---
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(docs)

# --- BM25 Setup ---
tokenized_docs = [doc.split() for doc in docs]
bm25 = BM25Okapi(tokenized_docs)

# Create a reusable structure
lexical_info = {
    "segments": segments,
    "docs": docs,
    "timestamps": timestamps,
    "tfidf_vectorizer": tfidf_vectorizer,
    "tfidf_matrix": tfidf_matrix,
    "bm25": bm25,
}

# Store a sample query for verification
query_sample = "What does the speaker say about the experiment results?"

# Prepare TF-IDF similarity
tfidf_query_vec = tfidf_vectorizer.transform([query_sample])
tfidf_similarities = (tfidf_matrix @ tfidf_query_vec.T).toarray().flatten()

# Prepare BM25 scores
bm25_scores = bm25.get_scores(query_sample.split())

# Collect top results
top_k = 5
tfidf_top_idx = np.argsort(tfidf_similarities)[::-1][:top_k]
bm25_top_idx = np.argsort(bm25_scores)[::-1][:top_k]

# Format results
tfidf_results = [(timestamps[i], docs[i], tfidf_similarities[i]) for i in tfidf_top_idx]
bm25_results = [(timestamps[i], docs[i], bm25_scores[i]) for i in bm25_top_idx]
print("TF-IDF Results:")
for timestamp, text, score in tfidf_results:
    print(f"Timestamp: {timestamp}, Text: {text}, Score: {score}")
print("\nBM25 Results:")
for timestamp, text, score in bm25_results:
    print(f"Timestamp: {timestamp}, Text: {text}, Score: {score}")
    


