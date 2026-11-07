from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from ..chunking import Chunk
from ..text import tokenize
from .base import RetrievedChunk


class BM25Retriever:
    """Okapi BM25 over character chunks."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._chunks = list(chunks)
        self._k1 = k1
        self._b = b
        self._tokens = [tokenize(chunk.text) for chunk in self._chunks]
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))
        total = len(self._tokens)
        self._average_length = sum(len(tokens) for tokens in self._tokens) / total if total else 0.0
        self._idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        query_terms = tokenize(query)
        if not query_terms or not self._chunks:
            return []
        scored: list[tuple[Chunk, float]] = []
        for chunk, tokens in zip(self._chunks, self._tokens, strict=True):
            counts = Counter(tokens)
            length = len(tokens)
            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if frequency == 0:
                    continue
                denominator = frequency + self._k1 * (
                    1 - self._b + self._b * length / max(self._average_length, 1e-9)
                )
                score += self._idf.get(term, 0.0) * frequency * (self._k1 + 1) / denominator
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [RetrievedChunk(chunk=chunk, score=score) for chunk, score in scored[:k]]
