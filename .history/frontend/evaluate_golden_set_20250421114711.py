import json
import time
import numpy as np
from collections import defaultdict


# --- Golden Dataset ---
answerable = [
    ("What is the topic of the PC seminar?", 0.00),
    ("Who is the professor presenting in the video?", 0.00),
    ("What is the token placement problem?", 737.28),
    ("What parameter is denoted by K in the context of token problems?", 1472.10),
    ("How many tokens are used per clique setup?", 2176.18),
    ("What is the condition on vertex degree in J for a trivial yes instance?", 2653.22),
    ("Why can’t blue edges exist inside J?", 2802.58),
    ("What is the required degree of a vertex in J to get a yes-instance?", 2653.22),
    ("What additional rule allows tokens to be removed or added while maintaining a minimum independent set size?", 3112.18),
    ("What is the equivalence established between token addition/removal and another token operation?", 3137.7)
]

unanswerable = [
    "What is the time complexity of the token sliding algorithm proposed in the seminar?",
    "Was the parameterized complexity of token deletion discussed in the seminar?",
    "Did the speaker mention any approximation algorithms for token problems?",
    "Which real-world application was used to illustrate the token jumping problem?",
    "Did the seminar explain how to handle vertices with negative degree in token problems?"
]

methods = ["faiss", "pgvector_ivfflat", "pgvector_hnsw", "lexical"]
TOLERANCE = 5.0

results = defaultdict(dict)

# --- Evaluation ---
for method in methods:
    correct = 0
    rejected = 0
    latencies = []

    for question, true_time in answerable:
        t0 = time.time()
        pred_time = get_top1_timestamp(question, method)
        t1 = time.time()
        latencies.append(t1 - t0)

        if pred_time is not None and abs(pred_time - true_time) <= TOLERANCE:
            correct += 1

    for question in unanswerable:
        pred_time = get_top1_timestamp(question, method)
        if pred_time is None:
            rejected += 1

    results[method]["accuracy"] = round(correct / len(answerable), 3)
    results[method]["rejection"] = round(rejected / len(unanswerable), 3)
    results[method]["latency"] = round(np.mean(latencies), 4)

# --- Save to file ---
with open("retrieval_eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Evaluation complete. Results saved to retrieval_eval_results.json.")
