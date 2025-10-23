import json
from typing import List, Dict, Any, Iterable
from pathlib import Path
from src.utils.text import normalize

def load_instances(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a list of instances in LongMemEval JSON")
    return data

def pick_question_ids(instances: List[Dict[str, Any]], desired: List[str]) -> List[str]:
    if desired:
        ids = set(desired)
        present = [x["question_id"] for x in instances if x.get("question_id") in ids]
        missing = ids - set(present)
        if missing:
            print(f"[warn] Missing question_ids (skipped): {sorted(missing)}")
        return present[:2]
    return [instances[0]["question_id"], instances[1]["question_id"]]

def turns_from_instance(inst: Dict[str, Any], granularity: str = "turn") -> Iterable[Dict[str, Any]]:
    qid = inst["question_id"]
    sessions = inst["haystack_sessions"]
    if granularity == "session":
        for s_idx, session in enumerate(sessions):
            parts = []
            for turn in session:
                role = turn.get("role", "")
                content = normalize(turn.get("content", ""))
                parts.append(f"{role}: {content}")
            yield {
                "doc_id": f"{qid}|s{s_idx}",
                "text": " ".join(parts).strip(),
                "meta": {"question_id": qid, "session_idx": s_idx},
            }
    else:
        for s_idx, session in enumerate(sessions):
            for t_idx, turn in enumerate(session):
                role = turn.get("role", "")
                content = normalize(turn.get("content", ""))
                yield {
                    "doc_id": f"{qid}|s{s_idx}|t{t_idx}",
                    "text": content,
                    "meta": {"question_id": qid, "session_idx": s_idx, "turn_idx": t_idx, "role": role},
                }
