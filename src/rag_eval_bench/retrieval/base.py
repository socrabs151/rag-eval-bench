from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..chunking import Chunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return the top-k chunks for a query, best first."""
        ...
