import os
import sys
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import time
import re
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
from nltk.corpus import stopwords
from data.Retrieval.faiss_search import run_retrieval as run_faiss
from data.Retrieval.pgvector_fusion_ivf import run_pgvector_ivfflat
from data.Retrieval.pgvector_fusion_hnsw import run_pgvector_hnsw
from data.Retrieval.lexical import run_lexical

Text_Model = "Salesforce/SFR-Embedding-Mistral"
Clip_Model = "openai/clip-vit-base-patch32"
device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(Text_Model)
text_model = AutoModel.from_pretrained(Text_Model).to(device)
clip_model = CLIPModel.from_pretrained(Clip_Model).to(device).eval()
clip_processor = CLIPProcessor.from_pretrained(Clip_Model)

stop_words = set(stopwords.words("english"))

def keyword_overlap(text, query, min_overlap=2):
    query_words = set(re.findall(r'\b\w+\b', query.lower())) - stop_words
    text_words = set(re.findall(r'\b\w+\b', text.lower()))
    return len(query_words & text_words) >= min_overlap

answerable = [
    ("Who is the professor presenting in the PC seminar?", 0.00, False),
    ("Show a numbered list in the slide.", 114.0, True),
    ("Show in the slide a graph", 2298.0, True),
    ("Show a slide with the mathematical proof", 3012.0, True),
    ("What parameter was used when studying token sliding and token jumping on bipartite graphs?", 1475.44, False),
    ("what degree must a vertex in graph J have if we have C3 and C4 freeness?", [2763.20, 2646.08], False),
    ("why blue edges can not exist in J?", 2800.56, False),
    ("What operation is allowed under the token addition and removal rule?", 3114.08, False),
    ("Is token sliding W1 hard when parameterized by the number of tokens?", 1516.80, False),
    ("Show a circle in the slide.", 3333.0, True),
]

unanswerable = [
    ("Show the professor while moving in the seminar.", True),
    ("Did the speaker mention any approximation algorithms for token problems?", False),
    ("Talk about a negative edge in a weighted graph.", False),
    ("Show the audience in the seminar.", True),
    ("What is the color of the token used in all token sliding reconfiguration proofs?", False),
]

thresholds = {
    "faiss": {"text": 0.5, "image": 0.26},
    "pgvector_ivfflat": {"text": 0.5, "image": 0.25},
    "pgvector_hnsw": {"text": 0.5, "image": 0.39},
    "lexical_tfidf": {"text": 1.0, "image": 0.0},
    "lexical_bm25": {"text": 1.0, "image": 0.0},
}

methods = list(thresholds.keys())
Tolerance = 5.0
Top_k = 5

results = defaultdict(dict)

for method in methods:
    correct = 0
    correct_text = 0
    correct_image = 0
    rejected = 0
    latencies = []
    text_thresh = thresholds[method]["text"]
    image_thresh = thresholds[method]["image"]

    for query, true_time, is_image in answerable:
        true_times = true_time if isinstance(true_time, list) else [true_time]
        t0 = time.time()
        if method.startswith("faiss"):
            transcript_results, image_results, _ = run_faiss(query, Top_k, tokenizer, text_model, clip_processor, clip_model, mode="Image" if is_image else "Text")
        elif method.startswith("pgvector_ivfflat"):
            transcript_results, image_results, _ = run_pgvector_ivfflat(query, Top_k, tokenizer, text_model, clip_processor, clip_model, mode="Image" if is_image else "Text")
        elif method.startswith("pgvector_hnsw"):
            transcript_results, image_results, _ = run_pgvector_hnsw(query, Top_k, tokenizer, text_model, clip_processor, clip_model, mode="Image" if is_image else "Text")
        elif method.startswith("lexical"):
            transcript_results = run_lexical(query, Top_k, method="tfidf" if "tfidf" in method else "bm25")
            image_results = []
        t1 = time.time()
        latencies.append(t1 - t0)

        if is_image and image_results:
            ts, _, score = image_results[0]
            if score >= image_thresh and any(abs(ts - t) <= Tolerance for t in true_times):
                correct += 1
                correct_image += 1
        elif not is_image and transcript_results:
            top_result = transcript_results[0]
            ts, text, score = top_result
            if score >= text_thresh and keyword_overlap(text, query) and any(abs(ts - t) <= Tolerance for t in true_times):
                correct += 1
                correct_text += 1

    for query, is_image in unanswerable:
        if method.startswith("faiss"):
            transcript_results, image_results, _ = run_faiss(query, Top_k, tokenizer, text_model, clip_processor, clip_model, mode="Image" if is_image else "Text")
        elif method.startswith("pgvector_ivfflat"):
            transcript_results, image_results, _ = run_pgvector_ivfflat(query, Top_k, tokenizer, text_model, clip_processor, clip_model, mode="Image" if is_image else "Text")
        elif method.startswith("pgvector_hnsw"):
            transcript_results, image_results, _ = run_pgvector_hnsw(query, Top_k, tokenizer, text_model, clip_processor, clip_model, mode="Image" if is_image else "Text")
        elif method.startswith("lexical"):
            transcript_results = run_lexical(query, Top_k, method="tfidf" if "tfidf" in method else "bm25")
            image_results = []
        if is_image:
            if not image_results or image_results[0][2] < image_thresh:
                rejected += 1
        else:
            if not transcript_results or transcript_results[0][2] < text_thresh or not keyword_overlap(transcript_results[0][1], query):
                rejected += 1

    if method.startswith("lexical"):
        total_answerable = sum(1 for _, _, is_image in answerable if not is_image)
        total_unanswerable = sum(1 for _, is_image in unanswerable if not is_image)
        total_text = total_answerable
        total_image = 0
    else:
        total_answerable = len(answerable)
        total_unanswerable = len(unanswerable)
        total_text = sum(1 for _, _, is_image in answerable if not is_image)
        total_image = sum(1 for _, _, is_image in answerable if is_image)

    results[method]["accuracy"] = round(correct / total_answerable, 3)
    results[method]["rejection"] = round(rejected / total_unanswerable, 3)
    results[method]["latency"] = round(np.mean(latencies), 4)
    results[method]["text_score"] = round(correct_text / total_text, 3) if total_text else 0.0
    results[method]["image_score"] = round(correct_image / total_image, 3) if total_image else 0.0

print("\n Retrieval Evaluation Summary:\n")
for method, stats in results.items():
    if method.startswith("lexical"):
        total_answerable = sum(1 for _, _, is_image in answerable if not is_image)
        total_unanswerable = sum(1 for _, is_image in unanswerable if not is_image)
        eval_scope = "Text only"
    else:
        total_answerable = len(answerable)
        total_unanswerable = len(unanswerable)
        eval_scope = "Text with Images"

    correct_a = int(stats["accuracy"] * total_answerable)
    correct_r = int(stats["rejection"] * total_unanswerable)

    print(f"Method: {method} [{eval_scope}]")
    print(f"  1. Accuracy on answerable:    {stats['accuracy']} ({correct_a} / {total_answerable})")
    print(f"  2. Rejection on unanswerable: {stats['rejection']} ({correct_r} / {total_unanswerable})")
    print(f"  3. Text-only accuracy:        {stats['text_score']}")
    print(f"  4. Image-only accuracy:       {stats['image_score']}")
    print(f"  5. Average Latency:           {stats['latency']} sec\n")

with open("retrieval_eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

