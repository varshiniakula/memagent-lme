🧠 LongMemEval – Multi-Retriever Evaluation for Memory-Based Agents
📘 Overview

This project evaluates how different retrievers (lexical, dense, ML-based, and time-aware) perform on the LongMemEval-S dataset — a benchmark for long-term conversational memory.

The goal is to understand how each retriever handles memory recall and to demonstrate that a multi-retriever orchestrator can outperform any single method when retrieving relevant sessions for long-context agents.

🚀 Key Contributions

Dataset Preparation:
Built and indexed the LongMemEval-S dataset into per-question Chroma collections, grouping all related chat sessions (with session_id and timestamps).

Retriever Evaluation Pipeline:
Implemented and ran multi-retriever experiments (BM25, TF-IDF, FAISS/KNN, SVM, Time-weighted) on 100 questions to measure accuracy@5 and mean recall@5.

Results Analysis & Visualization:
Created detailed reports, visual dashboards, and a mentor-ready one-pager proving that a multi-retriever orchestrator (BM25 → FAISS → Time-weighted → TF-IDF) achieves 100% coverage.
