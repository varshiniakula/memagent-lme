import argparse, json
from pathlib import Path
from typing import List, Dict, Any
import yaml
from tqdm import tqdm

from src.datasets.longmemeval_loader import load_instances, pick_question_ids, turns_from_instance
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.tfidf import TfidfRetriever
from src.retrievers.sbert import SBertFaissRetriever

def build_docs(instances: List[Dict[str, Any]], question_ids: List[str], granularity: str):
    idset = set(question_ids)
    subset = [x for x in instances if x["question_id"] in idset]
    queries = [{"question_id": x["question_id"], "question": x["question"]} for x in subset]
    docs = []
    for inst in subset:
        for d in turns_from_instance(inst, granularity=granularity):
            docs.append(d)
    return docs, queries

def get_retrievers(retrievers_cfg: str):
    cfg = yaml.safe_load(Path(retrievers_cfg).read_text())
    rets = []
    for name in cfg.get("sparse", []):
        if name == "bm25": rets.append(BM25Retriever())
        elif name == "tfidf": rets.append(TfidfRetriever())
    for m in cfg.get("dense_models", []):
        rets.append(SBertFaissRetriever(m))
    return rets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--granularity", choices=["turn", "session"], default="turn")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_file = cfg["data_file"]
    desired_ids = cfg.get("question_ids", [])
    retrievers_cfg = cfg["retrievers_config"]
    outdir = Path(cfg.get("output_dir", "outputs"))
    outdir.mkdir(parents=True, exist_ok=True)

    instances = load_instances(data_file)
    question_ids = pick_question_ids(instances, desired_ids)
    docs, queries = build_docs(instances, question_ids, args.granularity)

    rets = get_retrievers(retrievers_cfg)

    run_dir = outdir / f"retrieval_{args.granularity}"
    run_dir.mkdir(exist_ok=True, parents=True)

    for r in rets:
        print(f"[*] Building index for retriever: {r.name}")
        r.build(docs)
        out_path = run_dir / f"{r.name}.jsonl"
        with out_path.open("w", encoding="utf-8") as w:
            for q in tqdm(queries, desc=f"Querying {r.name}"):
                hits = r.query(q["question"], topk=args.topk)
                rec = {"question_id": q["question_id"], "question": q["question"], "retrieval_results": hits}
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[+] Wrote {out_path}")

if __name__ == "__main__":
    main()
