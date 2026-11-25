from __future__ import annotations

import pytest

from rag_eval_bench.chunking import Chunk
from rag_eval_bench.retrieval.base import RetrievedChunk
from rag_eval_bench.retrieval.rerank import CrossEncoderReranker


class FakeModel:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.calls: list[int] = []

    def predict(
        self, pairs: list[tuple[str, str]], batch_size: int = 32, show_progress_bar: bool = False
    ) -> list[float]:
        self.calls.append(len(pairs))
        assert batch_size > 0
        assert show_progress_bar is False
        return self._scores


def _hits(*ids: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=Chunk(id=f"{doc_id}#0", doc_id=doc_id, title=doc_id, text=f"text {doc_id}"),
            score=0.1,
        )
        for doc_id in ids
    ]


def test_rerank_orders_by_cross_encoder_score() -> None:
    model = FakeModel([0.2, 0.9, 0.5])
    reranker = CrossEncoderReranker("fake", model=model)

    result = reranker.rerank("query", _hits("a", "b", "c"), 2)

    assert [hit.chunk.doc_id for hit in result] == ["b", "c"]
    assert [hit.score for hit in result] == [0.9, 0.5]
    assert model.calls == [3]


def test_rerank_returns_everything_when_k_is_large() -> None:
    reranker = CrossEncoderReranker("fake", model=FakeModel([0.1, 0.4]))

    result = reranker.rerank("query", _hits("a", "b"), 10)

    assert [hit.chunk.doc_id for hit in result] == ["b", "a"]


def test_rerank_handles_empty_candidates() -> None:
    model = FakeModel([])
    reranker = CrossEncoderReranker("fake", model=model)

    assert reranker.rerank("query", [], 5) == []
    assert model.calls == []


def test_rerank_rejects_score_count_mismatch() -> None:
    reranker = CrossEncoderReranker("fake", model=FakeModel([0.5]))

    with pytest.raises(RuntimeError, match="scores"):
        reranker.rerank("query", _hits("a", "b"), 1)
