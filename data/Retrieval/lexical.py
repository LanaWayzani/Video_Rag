import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import re
import functools
from nltk.corpus import stopwords

stp_words = set(stopwords.words("english"))

def keyword_overlap(query_tokens, doc_text, min_overlap=2):
    doc_tokens = set(re_tokenize(doc_text))
    content_query_tokens = [t for t in query_tokens if t not in stp_words]
    return len(set(content_query_tokens) & doc_tokens) >= min_overlap

"""
This script loads preprocessed transcript segments and supports keyword-based retrieval
through two traditional methods: Term Frequency-Inverse Document Frequency (TF-IDF) and
BM25 scoring It returns the top-k relevant transcript segments for a given query.
"""

#Defining paths to data directories
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
seg_path = os.path.join(root, "data", "transcripts", "segments.json")
with open(seg_path, "r", encoding="utf-8") as f:
    segments = json.load(f)

docs = [seg["text"] for seg in segments]
timestamps = [seg["start"] for seg in segments]


def re_tokenize(text):
    """
    Lowercase and tokenize a text string into words using regex.
    Args:
        text (str): Input string
    Returns:
        List[str]: List of lowercase word tokens
    """
    return re.findall(r"\b\w+\b", text.lower())



@functools.lru_cache(maxsize=2)
def tfidf_model():
    """
    Fit a TF-IDF vectorizer and document-term matrix.
    Returns:
        Tuple[TfidfVectorizer, scipy.sparse matrix]: The vectorizer and its matrix
    """
    vectorizer = TfidfVectorizer(tokenizer=re_tokenize, lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(docs)
    return vectorizer, tfidf_matrix

@functools.lru_cache(maxsize=1)
def bm25_model():
    """
    Fit and cache a BM25 model over tokenized documents.
    Returns:
        BM25Okapi: Fitted BM25 index
    """
    tokenized_docs = [re_tokenize(doc) for doc in docs]
    return BM25Okapi(tokenized_docs)



def run_lexical(query, k=5, method="bm25"):
    """
    Perform lexical search using TF-IDF or BM25 and return top-k ranked results.

    Args:
        query (str): User query string
        k (int): Number of top results to return
        method (str): "tfidf" or "bm25" method to use

    Returns:
        List[Tuple[float, str, float]]: List of (timestamp, text, score) tuples
    """
    if not query.strip():
        return []

    query_tokens = re_tokenize(query)

    if method == "tfidf":
        vectorizer, tfidf_matrix = tfidf_model()
        tfidf_vec = vectorizer.transform([query])
        tfidf_scores = (tfidf_matrix @ tfidf_vec.T).toarray().flatten()
        tfidf_top_idx = np.argsort(tfidf_scores)[::-1][:k]
        top_score = tfidf_scores[tfidf_top_idx[0]] + 1e-9
        results = []
        for i in tfidf_top_idx:
            if keyword_overlap(query_tokens, docs[i]):
                results.append((timestamps[i], docs[i], round(tfidf_scores[i] / top_score, 4)))
        return results[:k]


    elif method == "bm25":
        bm25 = bm25_model()
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_top_idx = np.argsort(bm25_scores)[::-1][:k]
        top_score = bm25_scores[bm25_top_idx[0]] + 1e-9
        results = []
        for i in bm25_top_idx:
            if keyword_overlap(query_tokens, docs[i]):
                  results.append((timestamps[i], docs[i], round(bm25_scores[i] / top_score, 4)))
        return results[:k]


    else:
        raise ValueError("Unsupported method. Choose 'tfidf' or 'bm25'.")

"""
def main():
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Enter a text query: ").strip()
        if query.lower() == "exit":
            break

        method = input("Choose method [tfidf / bm25]: ").strip().lower()
        if method == "exit":
            break
        if method not in {"tfidf", "bm25"}:
            print("Invalid method. Please choose either 'tfidf' or 'bm25'.\n")
            continue

        k_input = input("How many top results?: ").strip()
        if k_input.lower() == "exit":
            break
        k = int(k_input) if k_input else 5

        print(f"\nRunning {method.upper()} retrieval for:\n\"{query}\"\n")
        results = run_lexical(query, k=k, method=method)

        if not results:
            print("No matches found.\n")
        else:
            for i, (timestamp, text, score) in enumerate(results, 1):
                print(f"[{i}] {timestamp:.2f}s | Score: {score:.4f}\n{text}\n")

        print("-" * 40 + "\n")


if __name__ == "__main__":
    main()
"""
