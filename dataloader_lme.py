# dataloader_lme.py
# Loader for LongMemEval cleaned datasets, aligned with the official spec:
# https://github.com/xiaowu0162/LongMemEval  (see README "Dataset Format")
# Fields used (per README):
#  - question_id, question_type, question, answer, question_date
#  - haystack_session_ids (parallel to haystack_sessions)
#  - haystack_dates          (parallel to haystack_sessions)
#  - haystack_sessions: list[ list[ {role, content, (optional) has_answer} ] ]
#  - answer_session_ids: list of session ids that contain the evidence
#
# We produce:
#   - documents: utterance-level LangChain Documents with metadata:
#       {sample_id, session_rank, turn_rank, dia_id, orig_session_id, role, date, question_id}
#   - qa_list: [{"sample_id","question","answer","evidence":[dia_ids],
#               "category_str","category"}]
#     Evidence is mapped primarily from turns with has_answer==True; if absent
#     we fall back to first turn of each session listed in answer_session_ids.
#
# Note: We do NOT invent IDs or parse weird strings; we follow the arrays as-is.

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

from langchain_core.documents import Document


# Optional: mapping if you still want integer categories in addition to the string label.
# You can ignore "category" downstream and use "category_str" instead if you prefer.
_QTYPE_TO_CAT = {
    # from README question_type set:
    # single-session-user, single-session-assistant, single-session-preference,
    # temporal-reasoning, knowledge-update, multi-session
    # If question_id ends with "_abs" it's an "abstention" question.
    "single-session-user": 1,
    "single-session-assistant": 1,
    "single-session-preference": 1,
    "temporal-reasoning": 2,
    "knowledge-update": 3,
    "multi-session": 4,
    "abstention": 5,
}

@dataclass
class LMEDataLoader:
    json_path: str

    def load_data(
            self,
            sample_ids: Optional[List[str]] = None,
            limit: Optional[int] = None,
    ) -> Tuple[List[Document], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Load LongMemEval cleaned JSON (S/M/oracle) and return (documents, qa_list, meta).
        """
        p = Path(self.json_path)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {self.json_path}")

        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Official files are lists of instances. (HuggingFace LFS pointers resolve to the JSON blob.)
        if not isinstance(data, list):
            raise ValueError("Expected a list of evaluation instances in LongMemEval JSON.")

        # Optionally filter & limit
        # inside LMEDataLoader.load_data(...)

        # Optionally filter & limit (robust to multiple types)
        # Acceptable forms for sample_ids:
        #   - None / "all" -> no filtering
        #   - list/tuple/set of IDs
        #   - comma-separated string: "q_0001,q_0007,42"
        #   - anything else -> ignored (with a warning)
        raw_sample_ids = sample_ids
        keep_ids = None

        if raw_sample_ids is None:
            keep_ids = None
        elif isinstance(raw_sample_ids, (list, tuple, set)):
            keep_ids = {str(x) for x in raw_sample_ids}
        elif isinstance(raw_sample_ids, str):
            if raw_sample_ids.strip().lower() in ("all", ""):
                keep_ids = None
            else:
                keep_ids = {s.strip() for s in raw_sample_ids.split(",") if s.strip()}
        else:
            # e.g., int by mistake — ignore gracefully
            print(f"[WARN] sample_ids has unexpected type {type(raw_sample_ids).__name__}; ignoring it.")
            keep_ids = None

        if keep_ids is not None:
            data = [x for x in data if str(x.get("question_id")) in keep_ids]

        if limit is not None:
            data = data[: int(limit)]


        documents, qa_list = self._normalize_longmemeval_cleaned(data)
        meta = {"total_instances": len(data), "source_file": str(self.json_path)}
        return documents, qa_list, meta

    # ---------------------------------------------------------------------

    def _normalize_longmemeval_cleaned(
            self, records: List[Dict[str, Any]]
    ) -> Tuple[List[Document], List[Dict[str, Any]]]:
        docs: List[Document] = []
        qas:  List[Dict[str, Any]] = []

        for idx, item in enumerate(records):
            qid   = str(item.get("question_id", f"q_{idx+1:04d}"))
            qtext = str(item.get("question", ""))
            ans   = str(item.get("answer", ""))
            qtype = str(item.get("question_type", "")) or ""
            qdate = item.get("question_date")

            # sessions aligned arrays
            sess_ids   = item.get("haystack_session_ids") or []   # original session IDs (strings/ints)
            sess_dates = item.get("haystack_dates") or []          # timestamps (optional)
            sessions   = item.get("haystack_sessions") or []       # list of sessions; each is a list of turns

            # Defensive checks
            if not isinstance(sessions, list):
                continue

            # Build a mapping from original session_id -> session_rank (1-based)
            # README: haystack_session_ids is parallel to haystack_sessions. :contentReference[oaicite:2]{index=2}
            sid_to_rank: Dict[str, int] = {}
            for s_rank, sid in enumerate(sess_ids, start=1):
                sid_to_rank[str(sid)] = s_rank

            # Build utterance-level documents
            # dia_id format we use: "S{rank}:{turn_rank}"
            for s_rank, session in enumerate(sessions, start=1):
                orig_sid = str(sess_ids[s_rank - 1]) if s_rank - 1 < len(sess_ids) else str(s_rank)
                date_val = sess_dates[s_rank - 1] if s_rank - 1 < len(sess_dates) else None

                if not isinstance(session, list):
                    continue

                for t_rank, turn in enumerate(session, start=1):
                    role = str(turn.get("role", "user"))
                    text = str(turn.get("content", ""))

                    dia_id = f"S{s_rank}:{t_rank}"

                    meta = {
                        "sample_id": qid,                # tie back to question instance
                        "question_id": qid,
                        "session_rank": s_rank,          # 1-based rank in the array
                        "turn_rank": t_rank,
                        "dia_id": dia_id,
                        "orig_session_id": orig_sid,     # original session id from file
                        "role": role,
                        "date": date_val,
                        "question_type": qtype,
                        "question_date": qdate,
                    }
                    docs.append(Document(page_content=text, metadata=meta))

            # --- Evidence mapping ---
            # Primary signal: turns that explicitly carry "has_answer: true".
            ev_turn_ids: List[str] = []
            for s_rank, session in enumerate(sessions, start=1):
                if not isinstance(session, list):
                    continue
                for t_rank, turn in enumerate(session, start=1):
                    if turn.get("has_answer", False) is True:
                        ev_turn_ids.append(f"S{s_rank}:{t_rank}")

            # Fallback: if no turn-level labels, use answer_session_ids at session granularity:
            # map each such session to the FIRST turn (rank 1) — consistent, explicit, and reproducible.
            if not ev_turn_ids:
                ans_sids = item.get("answer_session_ids") or []
                for sid in ans_sids:
                    s_rank = None
                    # sid should match an entry in haystack_session_ids (string/int)
                    if str(sid) in sid_to_rank:
                        s_rank = sid_to_rank[str(sid)]
                    elif isinstance(sid, int) and 1 <= sid <= len(sessions):
                        # some files may store rank directly, but README says ids; we still accept safe int ranks
                        s_rank = sid
                    if s_rank is not None and 1 <= s_rank <= len(sessions):
                        # first turn of that session
                        if isinstance(sessions[s_rank - 1], list) and len(sessions[s_rank - 1]) >= 1:
                            ev_turn_ids.append(f"S{s_rank}:1")

            # Compose QA entry
            cat_str = _normalize_qtype(qtype, qid)
            qa_entry = {
                "sample_id": qid,
                "question": qtext,
                "answer":   ans,
                "evidence": ev_turn_ids,     # list of dia_ids ("S{rank}:{turn}")
                "category_str": cat_str,     # original string
                "category": _QTYPE_TO_CAT.get(cat_str, 0),  # optional numeric bucket
            }
            qas.append(qa_entry)

        return docs, qas


def _normalize_qtype(qtype: str, qid: str) -> str:
    if qid.endswith("_abs"):
        return "abstention"
    return qtype or ""
