from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from ..text import tokenize

METRIC_DIRECTIONS = {
    "recall_at_5": "maximize",
    "mrr_at_10": "maximize",
    "f1": "maximize",
    "faithfulness": "maximize",
    "latency_ms": "minimize",
}


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if tokenize(prediction) == tokenize(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    predicted = tokenize(prediction)
    expected = tokenize(reference)
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    return 1.0 if set(retrieved_ids[:k]) & set(gold_ids) else 0.0


def reciprocal_rank(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    gold = set(gold_ids)
    for position, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold:
            return 1.0 / position
    return 0.0
