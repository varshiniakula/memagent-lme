#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import orjson
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PERSIST_DIR = "./chroma_longmemeval_s_subset"
DATA_PATH   = "./longmemeval_s_cleaned.json"
COLL_PREFIX = "vectordb_collection_"
MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE  = 64

def read_json(path: str) -> Any:
    p = Path(path)
    try:
        return orjson.loads(p.read_bytes())
    except Exception:
        return json.loads(p.read_text())

def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))

def parse_timestamp(s: str) -> str:
    # Input example: "2023/05/21 (Sun) 02:06" -> ISO
    try:
        return datetime.strptime(s, "%Y/%m/%d (%a) %H:%M").isoformat()
    except Exception:
        return s

def render_session_text(turns: List[Dict[str, str]]) -> str:
    lines = []
    for t in turns:
        role = (t.get("role") or "user").title()
        content = (t.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)

def choose_subset(rows, num, mode, seed, ids_csv):
    if ids_csv:
        want = set([x.strip() for x in ids_csv.split(",") if x.strip()])
        rows = [r for r in rows if r.get("question_id") in want]
        return rows[:num] if num else rows
    if mode == "random":
        import random
        rng = random.Random(seed)
        rows = rows[:]
        rng.shuffle(rows)
        return rows[:num]
    return rows[:num]

def ensure_collection(client, name: str, metadata: Dict[str, Any], overwrite: bool):
    # Create | overwrite | get
    if overwrite:
        try:
            client.delete_collection(name)
        except Exception:
            pass
        return client.create_collection(name=name, metadata=metadata)
    # else: reuse existing or create
    try:
        coll = client.get_collection(name)
        # Update collection-level metadata (Chroma 0.5 supports update_collection)
        try:
            client.update_collection(name=name, metadata=metadata)
        except Exception:
            pass
        return coll
    except Exception:
        return client.create_collection(name=name, metadata=metadata)

def safe_add(coll, ids: List[str], docs: List[str], metas: List[Dict[str, Any]], model, batch_size=BATCH_SIZE):
    """
    Add new points, skipping any IDs that already exist in the collection.
    (We compute embeddings only for the items that we actually add.)
    """
    assert len(ids) == len(docs) == len(metas)
    n = len(ids)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_ids = ids[start:end]

        # Check which of these IDs already exist
        try:
            got = coll.get(ids=batch_ids, include=["metadatas"])
            existing = set(got.get("ids", []))
        except Exception:
            existing = set()

        to_add_idx = [i for i, pid in enumerate(batch_ids) if pid not in existing]
        if not to_add_idx:
            continue

        add_docs  = [docs[start + i]  for i in to_add_idx]
        add_metas = [metas[start + i] for i in to_add_idx]
        add_ids   = [batch_ids[i]     for i in to_add_idx]

        embs = model.encode(add_docs, normalize_embeddings=True, show_progress_bar=False)
        coll.add(ids=add_ids, documents=add_docs, metadatas=add_metas, embeddings=embs)

def main(data_path: str, persist_dir: str, num: int, mode: str, seed: int,
         ids_csv: str, keep_conv_json: bool, overwrite: bool):
    os.makedirs(persist_dir, exist_ok=True)
    data = read_json(data_path)
    subset = choose_subset(data, num, mode, seed, ids_csv)

    client = chromadb.PersistentClient(path=persist_dir)
    model  = SentenceTransformer(MODEL_NAME)

    created = []
    for row in tqdm(subset, desc="Indexing selected questions"):
        qid   = row["question_id"]
        qtext = row.get("question", "")
        ans   = row.get("answer", "")
        qdate = row.get("question_date", "")

        sess_ids   = row["haystack_session_ids"]
        sess_dates = row["haystack_dates"]
        sessions   = row["haystack_sessions"]

        # Defensive: align lists
        assert len(sess_ids) == len(sess_dates) == len(sessions), f"Misaligned haystack lists for {qid}"

        # Deduplicate within this question (rare, but safe)
        uniq = {}
        for idx, (sid, sdate, turns) in enumerate(zip(sess_ids, sess_dates, sessions)):
            if sid not in uniq:
                uniq[sid] = (idx, sid, sdate, turns)
        ordered = [uniq[sid] for sid in uniq.keys()]

        # Prepare collection metadata (also store session→timestamp map as JSON string)
        sessions_map = [{"session_id": sid, "date": parse_timestamp(sdate)} for _, sid, sdate, _ in ordered]
        coll_meta = {
            "question_id": qid,
            "question": qtext,
            "answer": ans,
            "question_date": qdate,
            "num_sessions": len(sessions_map),
            "sessions_map_json": orjson.dumps(sessions_map).decode("utf-8"),
        }

        coll_name = f"{COLL_PREFIX}{qid}"
        coll = ensure_collection(client, coll_name, coll_meta, overwrite=overwrite)

        ids, docs, metas = [], [], []
        answer_session_ids = set(row.get("answer_session_ids", []))

        for local_idx, (idx, sid, sdate, turns) in enumerate(ordered):
            text = render_session_text(turns)
            ids.append(sid)  # keep document id == session_id for clean downstream
            meta = {
                "question_id": qid,
                "question": qtext,
                "answer": ans,
                "session_id": sid,
                "evidence": sid,
                "date": parse_timestamp(sdate),
                "turn_count": len(turns),
                "session_idx": local_idx,
                "is_answer_session": sid in answer_session_ids,
            }
            if keep_conv_json:
                meta["conversation_json"] = orjson.dumps(turns).decode("utf-8")  # string => Chroma-safe
            docs.append(text)
            metas.append(meta)

        safe_add(coll, ids, docs, metas, model, batch_size=BATCH_SIZE)

        created.append({
            "collection": coll_name,
            "question_id": qid,
            "question": qtext,
            "answer": ans,
            "num_sessions": len(ids),
            "session_ids": ids,
        })

    # update / write manifest
    manifest_path = Path(persist_dir) / "subset_manifest.json"
    write_json(manifest_path, {"persist_dir": persist_dir, "num_questions": len(created), "collections": created})
    print(f"\nWrote manifest: {manifest_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--persist_dir", default=PERSIST_DIR)
    ap.add_argument("--num", type=int, default=10)
    ap.add_argument("--mode", choices=["first","random"], default="first")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ids", type=str, default="", help="comma-separated question_ids")
    ap.add_argument("--keep_conv_json", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="drop & recreate a collection if it exists")
    args = ap.parse_args()
    main(args.data, args.persist_dir, args.num, args.mode, args.seed, args.ids, args.keep_conv_json, args.overwrite)
