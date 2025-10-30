# run_pipeline_lme.py
"""
Question-wise, multi-retriever runner for LongMemEval using RetrieverRegistry.

- Global index (robust)
- Retrievers via all_retrievers.RetrieverRegistry:
    bm25, tfidf, faiss, knn, svm, colbert, time_weighted, nanopq
    + multiquery, contextual_compression, self_query (LLM-enabled)

Outputs:
  results_lme/by_question_<ts>.json
  results_lme/summary_overall_<ts>.json

CLI (overrides config_lme.yaml):
  --json_path data/longmemeval_subset.json
  --limit 30
  --top_k 5
  --retrievers bm25,tfidf,faiss, multiquery,self_query,contextual_compression
  --sample_ids conv-0001,conv-0002

LLM (Google AI Studio):
  export GOOGLE_API_KEY="your-key"
  In config_lme.yaml:
    llm:
      provider: google
      model: "gemini-1.5-flash"
      temperature: 0
"""

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from langchain_core.documents import Document

from dataloader_lme import LMEDataLoader
from all_retrievers import RetrieverRegistry
from answer_evaluator import compare_evidence_ids, evaluate_retriever_results


# ---------------- Utils ----------------
def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sanitize_docs(docs: List[Document]) -> List[Document]:
    out = []
    for d in docs or []:
        txt = (d.page_content or "").strip()
        if txt:
            out.append(d)
    return out


# ------------- LLM bootstrap (Google) -------------
# --- LLM bootstrap (Google) with safe fallbacks -------------------------------
def build_llm_if_needed(config: Dict[str, Any], requested_retrievers: List[str]):
    """Return a Google Gemini LLM only if LLM retrievers are requested.
    Tries a small list of safe model fallbacks (adds '-latest' when needed)."""
    needs_llm = any(r in {"multiquery", "contextual_compression", "self_query"} for r in requested_retrievers)
    if not needs_llm:
        return None

    llm_cfg = (config or {}).get("llm", {}) or {}
    provider    = (llm_cfg.get("provider") or "google").lower()
    model_hint  = llm_cfg.get("model") or "gemini-2.5-flash"
    temperature = float(llm_cfg.get("temperature", 0))
    api_key_env = llm_cfg.get("api_key_env", "GOOGLE_API_KEY")

    if provider != "google":
        print(f"[WARN] LLM provider '{provider}' not supported in this script; using Google.")
        provider = "google"

    import os
    api_key = os.environ.get(api_key_env)
    if not api_key:
        print(f"[WARN] Google API key not found in env var '{api_key_env}'. "
              "LLM-based retrievers will be disabled for this run.")
        return None

    # Normalize model candidates
    candidates = []
    # Start with what user asked for
    candidates.append(model_hint)
    # If no '-latest' suffix, try with it
    if not model_hint.endswith("-latest"):
        candidates.append(f"{model_hint}-latest")
    # Common working fallbacks
    candidates.extend([
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash-8b-latest",
    ])
    # Deduplicate preserving order
    seen, norm = set(), []
    for m in candidates:
        if m and m not in seen:
            seen.add(m); norm.append(m)

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception as e:
        print(f"[WARN] langchain-google-genai not installed ({e}). LLM retrievers disabled.")
        return None

    # Try to construct the client. We don't do a paid 'ping' call here;
    # the retriever will call .invoke() later. If the model still 404s
    # at invoke-time, our registry will catch and skip that retriever.
    last_err = None
    for model in norm:
        try:
            llm = ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)
            print(f"[INFO] Using Google model: {model}")
            # Stash chosen model back into config (helps logging)
            config.setdefault("llm", {})["model"] = model
            return llm
        except Exception as e:
            last_err = e
            print(f"[WARN] Could not init Google model '{model}': {e}")
            continue

    print(f"[WARN] All Google model candidates failed: {norm}. Last error: {last_err}")
    print("[WARN] LLM-based retrievers will be disabled for this run.")
    return None



# ------------- Runner config -------------
@dataclass
class RunnerConfig:
    to_run: List[str]
    top_k: int


# ------------- Pipeline -------------
def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    # -------- Load data --------
    data_cfg = config.get("data") or {}
    json_path = data_cfg.get("json_path") or "data/longmemeval.json"
    print("[DEBUG] using data file:", json_path)
    if not Path(json_path).exists():
        raise FileNotFoundError(
            f"Data file not found at '{json_path}'. Set data.json_path or pass --json_path."
        )

    loader = LMEDataLoader(json_path=json_path)
    all_docs, qa_list, meta = loader.load_data(
        sample_ids=data_cfg.get("sample_ids"),
        limit=data_cfg.get("limit"),
    )
    all_docs = sanitize_docs(all_docs)
    if not all_docs:
        raise ValueError("No usable documents found after sanitization.")
    if not qa_list:
        raise ValueError("No questions found.")

    # Optional question limit (keep as-is; question-wise outputs anyway)
    q_limit = (config.get("evaluation") or {}).get("question_limit")
    if q_limit:
        qa_list = qa_list[: int(q_limit)]
    print(f"[DEBUG] loaded docs={len(all_docs)}, qas={len(qa_list)} (before filters)")

    # -------- Build retrievers --------
    ret_cfg = config.get("retrievers", {})
    to_run = [s.strip().lower() for s in (ret_cfg.get("to_run") or [])]
    if not to_run:
        to_run = ["bm25", "tfidf", "faiss"]  # sane default
    rcfg = RunnerConfig(
        to_run=to_run,
        top_k=int(ret_cfg.get("top_k", 5)),
    )

    # LLM only if requested
    llm = build_llm_if_needed(config, rcfg.to_run)

    registry = RetrieverRegistry(all_docs, config, llm=llm)
    available = registry.list_available()
    # Keep only retrievers that are actually available
    to_eval = [r for r in rcfg.to_run if r in available]
    if not to_eval:
        raise RuntimeError(f"No requested retrievers are available. Requested={rcfg.to_run} Available={available}")
    print("[DEBUG] available retrievers:", available)
    print("[DEBUG] evaluating retrievers :", to_eval)

    # -------- Evaluate (question-wise) --------
    detailed: List[Dict[str, Any]] = []
    tested = set()
    # Per-question winners view:
    by_question_rows: List[Dict[str, Any]] = []

    # Group QAs by sample for reporting order
    q_by_sample: Dict[str, List[Dict[str, Any]]] = {}
    for q in qa_list:
        q_by_sample.setdefault(q["sample_id"], []).append(q)

    for sid, qas in q_by_sample.items():
        for qa in qas:
            qtext = qa["question"]
            gold = qa.get("evidence", []) or []

            success_by: Dict[str, bool] = {}
            union_ids: List[str] = []
            winner_list: List[str] = []

            for name in to_eval:
                docs = registry.invoke(name, qtext, k=rcfg.top_k) or []
                metrics = compare_evidence_ids(gold, docs)
                # store detailed row too (for full trace)
                detailed.append({
                    "sample_id": sid,
                    "retriever": name,
                    "question": qtext,
                    "gold_evidence": gold,
                    "retrieved_dia_ids": [d.metadata.get("dia_id") for d in docs],
                    "metrics": metrics
                })
                tested.add(name)

                hit = metrics.get("evidence_found", 0) > 0
                success_by[name] = hit
                if docs:
                    union_ids.extend([d.metadata.get("dia_id") for d in docs if d.metadata.get("dia_id")])

            # winners = retrievers with at least one gold evidence
            winner_list = [r for r, ok in success_by.items() if ok]
            oracle_success = any(success_by.values())

            by_question_rows.append({
                "sample_id": sid,
                "question": qtext,
                "gold_evidence": gold,
                "success_by_retriever": success_by,
                "winner_list": winner_list,
                "oracle_success": oracle_success,
                "union_retrieved_dia_ids": list(dict.fromkeys(union_ids))[: rcfg.top_k * len(to_eval)]
            })

    # -------- Summaries --------
    overall = evaluate_retriever_results(detailed)

    # -------- Save --------
    out_cfg = config.get("output", {})
    out_dir = Path(out_cfg.get("results_dir", "./results_lme"))
    ensure_dir(out_dir)
    stamp = ts()

    byq_path = out_dir / f"by_question_{stamp}.json"
    overall_path = out_dir / f"summary_overall_{stamp}.json"
    with byq_path.open("w", encoding="utf-8") as f:
        json.dump({"timestamp": stamp, "questions": by_question_rows}, f, indent=2, ensure_ascii=False)
    with overall_path.open("w", encoding="utf-8") as f:
        json.dump({"timestamp": stamp, "retrievers_tested": sorted(tested), "summary": overall}, f, indent=2, ensure_ascii=False)

    # -------- Print compact table --------
    print("\nOVERALL")
    print("-" * 80)
    print(f"{'Retriever':<18} {'Questions':>10} {'Evidence Found':>18} {'Recall %':>10}")
    print("-" * 80)
    for r, s in sorted(overall.items()):
        qn = s.get("questions", 0); ef = s.get("evidence_found", 0); et = s.get("evidence_total", 1); rc = s.get("recall_pct", 0.0)
        print(f"{r:<18} {qn:>10} {f'{ef}/{et}':>18} {rc:>9.1f}%")
    print("-" * 80)

    print(f"\nByQuestion: {byq_path}")
    print(f"Overall   : {overall_path}")

    return {
        "by_question_path": str(byq_path),
        "overall_path": str(overall_path),
        "retrievers_tested": sorted(tested),
        "overall": overall,
    }


# ------------- CLI -------------
def parse_args():
    ap = argparse.ArgumentParser(description="Run LongMemEval pipeline (question-wise, multi-retriever).")
    ap.add_argument("--config", default="config_lme.yaml", help="Path to YAML config (optional).")
    ap.add_argument("--json_path", default=None, help="Dataset JSON path (overrides config).")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of conversations.")
    ap.add_argument("--top_k", type=int, default=None, help="Top-k for retrieval.")
    ap.add_argument("--retrievers", default=None, help="Comma-separated list of retrievers to run.")
    ap.add_argument("--sample_ids", default=None, help="Comma-separated conversation IDs to keep.")
    ap.add_argument("--llm_model", default=None, help="Override llm.model (e.g., gemini-1.5-flash-latest)")
    ap.add_argument("--llm_provider", default=None, help="Override llm.provider (google)")
    return ap.parse_args()


def main():
    args = parse_args()
    # Load config if present
    config_path = Path(args.config)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    # Merge CLI overrides
    config.setdefault("data", {})
    config.setdefault("retrievers", {})
    config.setdefault("output", {"results_dir": "./results_lme"})
    config.setdefault("evaluation", {})

    if args.json_path:
        config["data"]["json_path"] = args.json_path
    if args.limit is not None:
        config["data"]["limit"] = args.limit
    if args.sample_ids:
        config["data"]["sample_ids"] = [s.strip() for s in args.sample_ids.split(",") if s.strip()]
    if args.top_k is not None:
        config["retrievers"]["top_k"] = args.top_k
    if args.retrievers:
        config["retrievers"]["to_run"] = [s.strip() for s in args.retrievers.split(",") if s.strip()]
    if args.llm_model:
        config.setdefault("llm", {})["model"] = args.llm_model
    if args.llm_provider:
        config.setdefault("llm", {})["provider"] = args.llm_provider

    run_pipeline(config)


if __name__ == "__main__":
    main()
