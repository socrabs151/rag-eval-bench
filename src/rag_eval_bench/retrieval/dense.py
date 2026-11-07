from __future__ import annotations

from collections.abc import Sequence

from ..chunking import Chunk
from .base import RetrievedChunk


class DenseRetriever:
    """Bi-encoder retriever backed by sentence-transformers, imported lazily."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        model_name: str = "intfloat/multilingual-e5-small",
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Dense retrieval needs the optional dependency: pip install -e '.[retrieval]'"
            ) from error
        self._chunks = list(chunks)
        self._model = SentenceTransformer(model_name)
        self._vectors = self._model.encode(
            [chunk.text for chunk in self._chunks],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        vector = self._model.encode([query], normalize_embeddings=True)[0]
        scores = self._vectors @ vector
        order = scores.argsort()[::-1][:k]
        return [
            RetrievedChunk(chunk=self._chunks[index], score=float(scores[index])) for index in order
        ]
