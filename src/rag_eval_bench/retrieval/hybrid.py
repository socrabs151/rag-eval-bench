from __future__ import annotations

from collections.abc import Sequence

from ..chunking import Chunk
from .base import RetrievedChunk, Retriever


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]], *, k: int = 60
) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            chunk_id = hit.chunk.id
            chunks[chunk_id] = hit.chunk
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [RetrievedChunk(chunk=chunks[chunk_id], score=score) for chunk_id, score in ordered]


class HybridRetriever:
    """Reciprocal rank fusion over several retrievers."""

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        candidate_k: int = 50,
        rrf_k: int = 60,
    ) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        self._retrievers = list(retrievers)
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        rankings = [
            retriever.search(query, max(k, self._candidate_k)) for retriever in self._retrievers
        ]
        return reciprocal_rank_fusion(rankings, k=self._rrf_k)[:k]
