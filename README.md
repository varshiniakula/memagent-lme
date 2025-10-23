# memagent-lme

# MEMAGENT-LME — Retriever Comparison on LongMemEval 

A clean, modular system for comparing **13 retrievers** on long-context conversational QA using the **LongMemEval** dataset.

## 📋 Overview
- Load multi-session conversational data from LongMemEval (oracle/S/M cleaned files)
- Run 13 retrievers (2 sparse + 11 dense) with a consistent interface
- Save per-retriever JSONL outputs (top-k per question)
- Focused **2-conversation** evaluation to iterate quickly

## 🚀 Quick Start

### 1) Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
