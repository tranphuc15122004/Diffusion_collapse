"""Tier-1 factual knowledge and hallucination metrics (CoverageQA & PopQA)."""

from __future__ import annotations

import re
import string
from typing import Iterable, Sequence

REFUSAL_PATTERNS = (
    "i don't know",
    "i do not know",
    "i am not sure",
    "i'm not sure",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "no information",
    "not enough information",
    "as an ai",
)

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    text = text.lower().strip()
    text = text.translate(_PUNCT_TABLE)
    return " ".join(text.split())


def is_refusal(text: str) -> bool:
    norm = text.lower().strip()
    if not norm:
        return True
    return any(pat in norm for pat in REFUSAL_PATTERNS)


def clean_qa_prediction(raw_text: str) -> str:
    """Extract concise candidate answer span from a raw QA generation."""
    text = raw_text.strip()
    for prefix in ("answer:", "the answer is", "a:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix) :].strip()
    # Strip surrounding quotes/trailing period
    text = text.strip(" \t\n\"'`.,;:")
    return text


def _match_entity(pred_norm: str, alias: str, category: str | None = None) -> bool:
    alias_norm = normalize_text(alias)
    if not pred_norm or not alias_norm:
        return False
    if pred_norm == alias_norm:
        return True
    # SimpleStrat athlete rule: last-name match when single/full name matches last token
    if category == "athlete":
        return pred_norm.split()[-1] == alias_norm.split()[-1]
    # Substring boundary match when model outputs a short phrase like "the country is Brazil"
    pattern = rf"\b{re.escape(alias_norm)}\b"
    return re.search(pattern, pred_norm) is not None


def match_coverageqa_group(
    pred: str,
    answer_groups: Sequence[Sequence[str]],
    category: str | None = None,
) -> int | None:
    """Return index of the first matching CoverageQA equivalence group, or None."""
    pred_norm = normalize_text(clean_qa_prediction(pred))
    if not pred_norm or is_refusal(pred):
        return None
    for g_idx, group in enumerate(answer_groups):
        for alias in group:
            if _match_entity(pred_norm, alias, category=category):
                return g_idx
    return None


def compute_coverageqa_metrics(
    predictions: Sequence[str],
    answer_groups: Sequence[Sequence[str]],
    category: str | None = None,
) -> dict[str, float]:
    """Compute Recall@K, Precision@K, unique valid/invalid entity counts, and refusal rate."""
    if not answer_groups:
        raise ValueError("answer_groups must be non-empty")

    matched_groups: set[int] = set()
    unique_invalid: set[str] = set()
    refusals = 0

    for pred in predictions:
        cleaned = clean_qa_prediction(pred)
        if is_refusal(cleaned):
            refusals += 1
            continue
        g_idx = match_coverageqa_group(cleaned, answer_groups, category=category)
        if g_idx is not None:
            matched_groups.add(g_idx)
        else:
            norm = normalize_text(cleaned)
            if norm:
                unique_invalid.add(norm)

    n_valid = len(matched_groups)
    n_unique_pred = n_valid + len(unique_invalid)
    recall = n_valid / len(answer_groups)
    precision = (n_valid / n_unique_pred) if n_unique_pred > 0 else 0.0
    refusal_rate = refusals / len(predictions) if predictions else 0.0
    non_refusal_hallucination = (
        1.0 - precision if n_unique_pred > 0 else 0.0
    )
    return {
        "recall": float(recall),
        "precision": float(precision),
        "unique_valid": float(n_valid),
        "unique_total": float(n_unique_pred),
        "refusal_rate": float(refusal_rate),
        "hallucination_rate": float(non_refusal_hallucination),
    }


def match_popqa(pred: str, possible_answers: Iterable[str]) -> bool:
    pred_norm = normalize_text(clean_qa_prediction(pred))
    if not pred_norm or is_refusal(pred):
        return False
    return any(_match_entity(pred_norm, ans) for ans in possible_answers)


def compute_popqa_metrics(
    predictions: Sequence[str],
    possible_answers: Sequence[str],
) -> dict[str, float]:
    """Compute per-sample accuracy, pass@K, precision over non-refusals, and refusal rate."""
    if not predictions:
        return {"acc": 0.0, "pass_at_k": 0.0, "precision_non_refusal": 0.0, "hallucination_rate": 0.0, "refusal_rate": 0.0}

    correct = 0
    non_refusal = 0
    for pred in predictions:
        cleaned = clean_qa_prediction(pred)
        if is_refusal(cleaned):
            continue
        non_refusal += 1
        if match_popqa(cleaned, possible_answers):
            correct += 1

    acc = correct / len(predictions)
    prec_nr = (correct / non_refusal) if non_refusal > 0 else 0.0
    return {
        "acc": float(acc),
        "pass_at_k": 1.0 if correct > 0 else 0.0,
        "precision_non_refusal": float(prec_nr),
        "hallucination_rate": float(1.0 - prec_nr) if non_refusal > 0 else 0.0,
        "refusal_rate": float((len(predictions) - non_refusal) / len(predictions)),
    }


def compute_thr(
    acc_tail_instruct: float,
    acc_tail_base: float,
    acc_head_instruct: float,
    acc_head_base: float,
    eps: float = 1e-8,
) -> float:
    """Tail-to-Head Retention Ratio: (Acc_Tail(Inst)/Acc_Tail(Base)) / (Acc_Head(Inst)/Acc_Head(Base))."""
    tail_ratio = acc_tail_instruct / max(acc_tail_base, eps)
    head_ratio = acc_head_instruct / max(acc_head_base, eps)
    return float(tail_ratio / max(head_ratio, eps))
