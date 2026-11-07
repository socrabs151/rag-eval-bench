import pytest

from rag_eval_bench.evaluation.metrics import (
    exact_match,
    recall_at_k,
    reciprocal_rank,
    token_f1,
)


def test_token_f1_on_identical_text() -> None:
    assert token_f1("A retriever and a generator.", "a retriever and a generator") == 1.0


def test_token_f1_on_partial_overlap() -> None:
    assert token_f1("retriever", "a retriever and a generator") == pytest.approx(1 / 3)


def test_token_f1_on_disjoint_text() -> None:
    assert token_f1("bm25", "dense") == 0.0


def test_exact_match_normalizes_case_and_punctuation() -> None:
    assert exact_match("A Retriever!", "a retriever") == 1.0
    assert exact_match("retriever", "generator") == 0.0


def test_recall_at_k_respects_the_cut() -> None:
    assert recall_at_k(["a", "b"], ["b"], 2) == 1.0
    assert recall_at_k(["a", "b"], ["b"], 1) == 0.0
    assert recall_at_k(["a"], ["z"], 5) == 0.0


def test_reciprocal_rank_counts_the_first_relevant_position() -> None:
    assert reciprocal_rank(["x", "gold"], ["gold"]) == 0.5
    assert reciprocal_rank(["gold", "x"], ["gold"]) == 1.0
    assert reciprocal_rank(["x"], ["gold"]) == 0.0
