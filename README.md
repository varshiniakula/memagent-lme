# 🧠 MEMAGENT-LME (LongMemEval Retriever Evaluation)

A modular, category-aware benchmark for **retrieval-based conversational QA** on the **LongMemEval dataset**.
---

## 📋 Overview

The project benchmarks various retrievers (lexical, vector, hybrid, and ML-based) on **LongMemEval** — a conversational dataset designed to test **long-context memory** and **retrieval accuracy** in Q&A tasks.

It enables:

1. Loading and normalizing **LongMemEval conversations** (`turns`, `qa`, `evidence`).
2. Running multiple retrievers on Q&A pairs.
3. Evaluating **recall** of ground-truth evidence.
4. Categorizing questions (e.g., Factual, Temporal, Reasoning).
5. Comparing retriever performance **by question category**.

---

## 🚀 Quick Start

### 1️⃣ Setup

```bash
git clone <this_repo>
cd memagent-lme
python -m venv .venv
source .venv/bin/activate       # or .venv\Scripts\activate (Windows)
pip install -r requirements.txt
```

### 2️⃣ Prepare the Dataset

Run the dataset extender to generate or merge conversations:

```bash
python tools/extend_lme_dataset.py --auto-categorize

# (Optional) Merge additional JSONs and auto-categorize
python tools/extend_lme_dataset.py --merge path/to/raw_lme.json --auto_categorize
```

Check your dataset:

```bash
python tools/inspect_categories.py

### 3️⃣ Configure

Edit `config_lme.yaml`:

```yaml
data:
  json_path: "data/longmemeval.json"
  sample_ids: "all"
  limit: null

retrievers:
  top_k: 5
  embedding_model: "sentence-transformers/all-mpnet-base-v2"

evaluation:
  question_limit: null

output:
  results_dir: "./results_lme"
  timestamp_format: "datetime"
```

---

### 4️⃣ Run the Category-Aware Pipeline

```bash
python run_pipeline_lme.py
```

Outputs saved to `results_lme/`:

| File                         | Description                     |
| ---------------------------- | ------------------------------- |
| `retriever_all_*.json`       | Detailed per-question results   |
| `summary_overall_*.json`     | Overall recall per retriever    |
| `summary_by_category_*.json` | Recall per retriever × category |


## 🧩 Supported Retrievers

| Retriever                          | Type               | Description                  |
| ---------------------------------- | ------------------ | ---------------------------- |
| **BM25**                           | Keyword            | Classic lexical ranking      |
| **TF-IDF**                         | Keyword            | Traditional IR baseline      |
| **FAISS (KNN)**                    | Vector             | Dense semantic similarity    |
| **SVM**                            | ML                 | Embedding-based re-ranker    |
| **Ensemble**                       | Hybrid             | FAISS + BM25 weighted fusion |
| **Time-weighted**                  | Temporal           | Recency-aware variant        |
| **(Optional)** RAGatouille-ColBERT | LLM Dense          | Contextual LLM retrieval     |
| **(Optional)** NanoPQ              | Vector Compression | Efficient large-scale FAISS  |

---

## 🧠 Categories (Mirrors Locomo’s 5)

| Category | Type                           | Description                          | Example                                |
| -------- | ------------------------------ | ------------------------------------ | -------------------------------------- |
| **1**    | Multi-Part Factual             | Answers with multiple discrete facts | “List the city and country.”           |
| **2**    | Temporal / Time-based          | Requires time or event recall        | “When is the meeting scheduled?”       |
| **3**    | Inference / Reasoning          | Needs reasoning beyond surface text  | “Would they still attend if…”          |
| **4**    | Interpretation / Understanding | Explains intent or emotion           | “Why did she feel inspired?”           |
| **5**    | Adversarial / No-Answer        | Unanswerable or misleading           | “What’s the speed of light in dreams?” |

Current pipeline evaluates **Categories 1 & 2** (approachable via RAG).

---

## ⚙️ Tools

| Tool                          | Path                                            | Function |
| ----------------------------- | ----------------------------------------------- | -------- |
| `tools/extend_lme_dataset.py` | Add or merge conversations, auto-categorize QAs |          |
| `tools/inspect_categories.py` | Inspect dataset size and category distribution  |          |
| `dataloader_lme.py`           | Load & normalize conversation data              |          |
| `run_pipeline_lme.py`         | Core benchmark runner                           |          |
| `answer_evaluator.py`         | Evaluate recall vs ground-truth evidence        |          |

---



