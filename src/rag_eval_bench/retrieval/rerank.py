from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from .base import RetrievedChunk

DEFAULT_BATCH_SIZE = 32

RERANKER_MODELS: dict[str, str] = {
    "ms-marco-mini": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge-base": "BAAI/bge-reranker-base",
}


class Reranker(Protocol):
    def rerank(self, query: str, hits: Sequence[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        """Re-score candidates for a query and return the best k."""
        ...


class CrossEncoderReranker:
    """Cross-encoder reranker, imported lazily so the base install stays small."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        model: Any | None = None,
        max_length: int = 512,
    ) -> None:
        if model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as error:
                raise RuntimeError(
                    "Reranking needs the optional dependency: pip install -e '.[retrieval]'"
                ) from error
            model = CrossEncoder(model_name, max_length=max_length)
        self.model_name = model_name
        self._model = model
        self._batch_size = batch_size

    def rerank(self, query: str, hits: Sequence[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        if not hits or k <= 0:
            return []
        pairs = [(query, hit.chunk.text) for hit in hits]
        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        if len(scores) != len(hits):
            raise RuntimeError(f"reranker returned {len(scores)} scores for {len(hits)} candidates")
        order = sorted(range(len(hits)), key=lambda index: float(scores[index]), reverse=True)
        return [
            RetrievedChunk(chunk=hits[index].chunk, score=float(scores[index]))
            for index in order[:k]
        ]
