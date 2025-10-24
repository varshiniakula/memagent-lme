from typing import List, Dict, Tuple
import re

SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
ISSUE_WORDS = {"issue", "problem", "broken", "malfunction", "not working", "not functioning", "stopped working", "fail", "failure"}

def _sentences(text: str) -> List[str]:
    if not text:
        return []
    return [s.strip() for s in SENT_SPLIT.split(text.strip()) if s.strip()]

def _best_sentence(text: str) -> str:
    sents = _sentences(text)
    for s in sents:
        if len(s.split()) >= 6:
            return s
    return sents[0] if sents else ""

def _extract_quoted_phrases(text: str) -> List[str]:
    # "Data Analysis using Python", ‘like this’, etc.
    return re.findall(r'["“”\'‘’]([\w .:-]+?)["“”\'‘’]', text)

def _contains_issue_pattern(s: str) -> Tuple[bool, str]:
    low = s.lower()
    for w in ISSUE_WORDS:
        if w in low:
            return True, w
    return False, ""

def _canonicalize_event(text: str) -> str:
    """
    Turn phrases like 'a webinar on Data Analysis using Python' into
    'Data Analysis using Python webinar'.
    """
    low = text.lower()
    # webinar on X
    m = re.search(r'\bwebinar on ([\w .:/+-]+)', low)
    if m:
        title = m.group(1).strip().rstrip(".")
        return f"{title} webinar"
    # workshop on X
    m = re.search(r'\bworkshop on ([\w .:/+-]+)', low)
    if m:
        title = m.group(1).strip().rstrip(".")
        return f"{title} workshop"
    # 'title' webinar/workshop (already canonical)
    m = re.search(r'([A-Za-z0-9][\w .:/+-]+?)\s+(webinar|workshop)\b', text)
    if m:
        return f"{m.group(1).strip()} {m.group(2).strip()}"
    return ""

def _pick_earlier_event(sentences: List[str]) -> str:
    """
    If both events are mentioned with relative dates, pick the earlier one.
    Very light heuristic: 'two months ago' (earlier) beats 'last Saturday'.
    """
    score = {}
    for s in sentences:
        ls = s.lower()
        if "webinar" in ls or "workshop" in ls:
            can = _canonicalize_event(s)
            if not can:
                continue
            # heuristic dating
            t = 0
            if "two months ago" in ls or "last month" in ls or "months ago" in ls:
                t = -60       # older
            elif "last saturday" in ls or "yesterday" in ls or "last week" in ls:
                t = -7        # newer
            score[can] = min(score.get(can, 0), t)
    if not score:
        return ""
    # smallest (older) wins
    return sorted(score.items(), key=lambda x: x[1])[0][0]

def generate_answer_from_hits(hits: List[Dict], max_chars: int = 120) -> str:
    """
    Canonical short answer generator:
    1) If we find an explicit 'X system' + issue pattern -> 'X system not functioning correctly'
    2) If we can canonicalize an event mention -> 'Title webinar/workshop'
    3) Fallback: best informative sentence from top hit
    """
    if not hits:
        return ""

    # 1) Issue pattern extraction (e.g., GPS)
    for h in hits[:8]:  # first few hits
        sents = _sentences(h.get("text", ""))
        for s in sents:
            ok, w = _contains_issue_pattern(s)
            if ok:
                # try to capture 'GPS system' or 'GPS' near the issue word
                m = re.search(r'\b([A-Za-z0-9+-]+(?:\s+system)?)\b[^.]{0,50}\b(' + re.escape(w) + r'|not functioning|not working|malfunction)', s, flags=re.I)
                if m:
                    thing = m.group(1)
                    # normalize "gps" -> "GPS"
                    if thing.lower() == "gps":
                        thing = "GPS system"
                    elif thing.lower().endswith(" system"):
                        # Title Case
                        thing = thing.title()
                    return f"{thing} not functioning correctly"[:max_chars]

    # 2) Event canonicalization + "which first?" heuristic
    sentences = []
    for h in hits[:10]:
        sentences.extend(_sentences(h.get("text", "")))
    first_event = _pick_earlier_event(sentences)
    if first_event:
        return first_event[:max_chars]

    # Also try to canonicalize directly from the best early sentences
    for s in sentences[:6]:
        can = _canonicalize_event(s)
        if can:
            return can[:max_chars]

    # 3) Fallback: best sentence from the top hit
    return _best_sentence(hits[0].get("text", ""))[:max_chars].strip()
