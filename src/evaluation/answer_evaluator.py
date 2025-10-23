"""
Answer Evaluator for MEMAGENT-LME
---------------------------------
Compares generated answers with ground truth for conversational QA tasks.
Supports substring match (Locomo style) and optional token F1.
"""

from typing import Dict
import re
from collections import Counter


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse spaces, strip punctuation."""
    if not text:
        return ""
    text = str(text).lower().strip()
    text = " ".join(text.split())
    text = re.sub(r"[.,!?;:]", "", text)
    return text


def evaluate_answer(generated: str, ground_truth: str) -> Dict[str, bool]:
    if not generated:
        return {"is_correct": False, "has_answer": False}
    if not ground_truth:
        return {"is_correct": False, "has_answer": True}

    gen_norm = normalize_text(generated)
    gt_norm  = normalize_text(ground_truth)

    # strict (Locomo style)
    strict = (gt_norm in gen_norm) or (gen_norm in gt_norm)

    # soft: token F1 >= 0.5 counts as correct
    soft = f1_score(generated, ground_truth) >= 0.5

    return {"is_correct": bool(strict or soft), "has_answer": True}



def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 for extra signal (optional)."""
    p = normalize_text(prediction).split()
    g = normalize_text(ground_truth).split()
    if not p or not g:
        return 1.0 if p == g else 0.0
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall    = num_same / len(g)
    return 2 * precision * recall / (precision + recall)
