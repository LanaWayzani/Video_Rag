# Video Retrieval-Augmented Generation (RAG) System

This repository contains a complete implementation of a **Video Retrieval-Augmented Generation (RAG) System** that enables semantic and lexical retrieval from video transcripts and visual keyframes. The system leverages a multimodal architecture, embedding text queries using the "Salesforce/SFR-Embedding-Mistral" model and image features using "openai/clip-vit-base-patch32".
Retrieval can be performed using several backend options, including in-memory vector search via **FAISS**, PostgreSQL-based vector search using **pgvector** with both "IVFFLAT" and "HNSW" indexing strategies, and traditional lexical methods like **TF-IDF** and **BM25**.

Users can submit queries using textual, visual, or fused modalities. The interface is implemented in **Streamlit** and offers an interactive sidebar to choose query types, retrieval backends, top-k results, fusion weights, and similarity thresholds. The system is enhanced with a set of robust filtering mechanisms, including lexical keyword overlap checks, vague query rejection, low-score suppression, and similarity-based filtering, ensuring that only meaningful and relevant matches are displayed. Each result is highlighted with matched terms, time-aligned keyframes, and embedded video playback around the top result timestamp.

An evaluation script is included, which benchmarks each retrieval method using a curated golden dataset of answerable and unanswerable queries. It computes metrics such as **accuracy**, **rejection rate**, **average latency**, and **per-modality performance scores**. The reported metrics reflect the optimized performance of each backend after applying all filtering and scoring improvements.

**Note**:  
1. PostgreSQL credentials must be provided in a local ".env" file .  
2. Video files ( "complexity.mp4") are **not included** in the repository. These must be stored and accessed locally for full functionality.  
3. All Python dependencies are listed in "requirements.txt".


