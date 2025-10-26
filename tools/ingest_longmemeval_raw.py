# tools/ingest_longmemeval_raw.py
"""
Convert the *raw* LongMemEval dataset into the internal format expected by our runner:
[
  {
    "id": "conv-XXXX",
    "turns": [{"speaker": "A/B/...","text": "...","dia_id": "T{id}:{local_turn_index}"}],
    "qa": [{"question": "...","answer": "...","evidence": ["T{id}:{local_turn_index}", ...]}]
  },
  ...
]

Usage:
  python tools/ingest_longmemeval_raw.py --src ./path/to/raw --out data/longmemeval_raw.json

The script is schema-tolerant:
- If --src is a file: reads it (json or jsonl) and tries to infer fields.
- If --src is a directory: looks for common names like conversations.jsonl / dialogue.jsonl / qa.jsonl.
- Tries several key aliases for turns and QAs. If evidence is missing, leaves [].
"""
import json, os, sys, argparse, glob

TURN_KEYS = ["turns", "dialogue", "dialog", "conversation"]
TURN_ITEM_SPEAKER_KEYS = ["speaker", "role", "from"]
TURN_ITEM_TEXT_KEYS = ["text", "content", "utterance"]

QA_KEYS = ["qa", "qas", "questions"]
QA_ITEM_QUESTION_KEYS = ["question", "query", "q"]
QA_ITEM_ANSWER_KEYS = ["answer", "a", "gold_answer"]
QA_ITEM_EVIDENCE_KEYS = ["evidence", "gold_evidence", "gold_utterances", "gold_utts", "gold"]

def _read_json_or_jsonl(p):
    if p.endswith(".jsonl"):
        with open(p, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    else:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            # normalize top-level: either list or {data:[...]}
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                return data["data"]
            return data

def _find_first_key(d, keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default

def _find_path_candidates(src_dir):
    # Common raw distributions
    pats = [
        "conversations.jsonl", "dialogue.jsonl", "dialog.jsonl",
        "conversations.json", "dialogue.json", "dialog.json",
        "qa.jsonl", "qa.json", "questions.jsonl", "questions.json"
    ]
    hits = {}
    for pat in pats:
        for p in glob.glob(os.path.join(src_dir, pat)):
            hits[os.path.basename(p)] = p
    return hits

def _normalize_conv(i, raw_item):
    # id
    cid = str(raw_item.get("id") or raw_item.get("conversation_id") or f"conv-{i+1:04d}")

    # turns
    raw_turns = _find_first_key(raw_item, TURN_KEYS, default=[])
    if not raw_turns and "utterances" in raw_item:
        raw_turns = raw_item["utterances"]

    turns = []
    for j, t in enumerate(raw_turns):
        spk = _find_first_key(t, TURN_ITEM_SPEAKER_KEYS, default="A")
        txt = _find_first_key(t, TURN_ITEM_TEXT_KEYS, default="")
        dia_id = t.get("dia_id") or f"T{cid.split('-')[-1]}:{j+1}"
        turns.append({"speaker": spk, "text": txt, "dia_id": dia_id})

    # qa
    raw_qas = _find_first_key(raw_item, QA_KEYS, default=[])
    qas = []
    for q in raw_qas:
        qtext = _find_first_key(q, QA_ITEM_QUESTION_KEYS, default="")
        ans = _find_first_key(q, QA_ITEM_ANSWER_KEYS, default="")
        ev = _find_first_key(q, QA_ITEM_EVIDENCE_KEYS, default=[])
        # normalize evidence to list of dia_ids (strings)
        if isinstance(ev, (str, int)):
            ev = [str(ev)]
        ev = [str(x) for x in ev]
        qas.append({"question": qtext, "answer": ans, "evidence": ev, "category": None})

    return {"id": cid, "turns": turns, "qa": qas}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to raw dataset (file or directory)")
    ap.add_argument("--out", default="data/longmemeval_raw.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if os.path.isdir(args.src):
        hits = _find_path_candidates(args.src)
        # try to load a single combined file if present
        combined = None
        for name in ["conversations.json", "conversations.jsonl", "dialogue.json", "dialogue.jsonl", "dialog.json", "dialog.jsonl"]:
            if name in hits:
                combined = _read_json_or_jsonl(hits[name])
                break
        if combined is None:
            # fall back: concatenate anything we have
            combined = []
            for p in hits.values():
                combined.extend(_read_json_or_jsonl(p))
        raw_list = combined
    else:
        raw_list = _read_json_or_jsonl(args.src)

    if not isinstance(raw_list, list):
        print("ERROR: Expected a list at top-level. Got:", type(raw_list), file=sys.stderr)
        sys.exit(1)

    out = []
    for i, item in enumerate(raw_list):
        try:
            out.append(_normalize_conv(i, item))
        except Exception as e:
            print(f"[WARN] Failed to normalize item {i}: {e}", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote {len(out)} conversations to {args.out}")

if __name__ == "__main__":
    main()
