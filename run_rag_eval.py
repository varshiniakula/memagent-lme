import argparse, json
from pathlib import Path
from typing import List, Dict, Any
import yaml
from tqdm import tqdm

from src.datasets.longmemeval_loader import load_instances, pick_question_ids, turns_from_instance
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.tfidf import TfidfRetriever
from src.retrievers.sbert import SBertFaissRetriever
from src.generation.answerers import generate_answer_from_hits
from src.evaluation.answer_evaluator import evaluate_answer, f1_score


def build_docs(instances: List[Dict[str, Any]], question_ids: List[str], granularity: str):
    idset = set(question_ids)
    subset = [x for x in instances if x["question_id"] in idset]
    queries = [{"question_id": x["question_id"], "question": x["question"]} for x in subset]
    docs = []
    for inst in subset:
        for d in turns_from_instance(inst, granularity=granularity):
            docs.append(d)
    return docs, queries, subset


def get_retrievers(retrievers_cfg: str):
    cfg = yaml.safe_load(Path(retrievers_cfg).read_text())
    rets = []
    for name in cfg.get("sparse", []):
        if name == "bm25":
            rets.append(BM25Retriever())
        elif name == "tfidf":
            rets.append(TfidfRetriever())
    for m in cfg.get("dense_models", []):
        rets.append(SBertFaissRetriever(m))
    return rets


def get_ground_truth(inst: Dict[str, Any]) -> str:
    # LongMemEval fields vary by release; try common candidates.
    for k in ["answer", "final_answer", "ground_truth", "gold_answer", "target", "reference_answer"]:
        v = inst.get(k)
        if isinstance(v, str) and v.strip():
            return v
    # Some variants store answers inside structure; add custom logic if needed.
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Same YAML as run_retrieval.py (data_file, question_ids, ...)")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--granularity", choices=["turn", "session"], default="turn")
    ap.add_argument("--max_answer_chars", type=int, default=400)
    ap.add_argument("--out_dir", default="outputs_rag")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_file = cfg["data_file"]
    desired_ids = cfg.get("question_ids", [])
    retrievers_cfg = cfg["retrievers_config"]

    instances = load_instances(data_file)
    # Use same two-question selection
    question_ids = pick_question_ids(instances, desired_ids)
    docs, queries, subset = build_docs(instances, question_ids, args.granularity)
    qid2inst = {x["question_id"]: x for x in subset}

    rets = get_retrievers(retrievers_cfg)

    out_dir = Path(args.out_dir)
    ans_dir = out_dir / f"answers_{args.granularity}"
    sum_dir = out_dir / f"summaries_{args.granularity}"
    ans_dir.mkdir(parents=True, exist_ok=True)
    sum_dir.mkdir(parents=True, exist_ok=True)

    for r in rets:
        print(f"[*] Building index for retriever: {r.name}")
        r.build(docs)

        # Write per-question answers (for traceability)
        answers_path = ans_dir / f"{r.name}.jsonl"
        summary_path = sum_dir / f"{r.name}_summary.json"

        results = []
        with answers_path.open("w", encoding="utf-8") as w:
            for q in tqdm(queries, desc=f"RAG {r.name}"):
                hits = r.query(q["question"], topk=args.topk)
                pred_answer = generate_answer_from_hits(hits, max_chars=args.max_answer_chars)
                gt_answer = get_ground_truth(qid2inst[q["question_id"]] )

                eval_res = evaluate_answer(pred_answer, gt_answer)
                f1 = f1_score(pred_answer, gt_answer) if gt_answer else 0.0

                line = {
                    "question_id": q["question_id"],
                    "question": q["question"],
                    "pred_answer": pred_answer,
                    "gold_answer": gt_answer,
                    "eval": {**eval_res, "f1": f1},
                    "retrieval_results": hits,
                }
                w.write(json.dumps(line, ensure_ascii=False) + "\n")
                results.append(line)

        # Summarize
        total = len(results)
        correct = sum(1 for x in results if x["eval"]["is_correct"])
        answered = sum(1 for x in results if x["eval"]["has_answer"])
        mean_f1 = sum(x["eval"]["f1"] for x in results) / max(total, 1)

        summary = {
            "retriever": r.name,
            "granularity": args.granularity,
            "topk": args.topk,
            "questions": total,
            "answered_pct": 100.0 * answered / max(total, 1),
            "is_correct_pct": 100.0 * correct / max(total, 1),
            "mean_f1": mean_f1,
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"[+] Answers -> {answers_path}")
        print(f"[+] Summary -> {summary_path}")
        print(
            f"    answered%={summary['answered_pct']:.1f}  "
            f"is_correct%={summary['is_correct_pct']:.1f}  "
            f"mean F1={summary['mean_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
