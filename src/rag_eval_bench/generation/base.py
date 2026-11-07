from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..retrieval.base import RetrievedChunk


@dataclass(frozen=True)
class Generation:
    text: str
    latency_ms: int
    tokens: int


class Generator(Protocol):
    name: str

    def generate(self, question: str, contexts: Sequence[RetrievedChunk]) -> Generation:
        """Answer the question from the retrieved context."""
        ...


def build_prompt(question: str, contexts: Sequence[RetrievedChunk]) -> str:
    blocks = [
        f"[{index}] {hit.chunk.title}: {hit.chunk.text}"
        for index, hit in enumerate(contexts, start=1)
    ]
    context_text = "\n".join(blocks) if blocks else "No context retrieved."
    return (
        "Answer the question using only the context below. "
        "Cite sources as [n]. If the context does not contain the answer, say you do not know.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    )
