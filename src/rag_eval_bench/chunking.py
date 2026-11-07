from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .schemas import CorpusDoc


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    title: str
    text: str


def chunk_documents(
    docs: Sequence[CorpusDoc], *, size: int = 400, overlap: int = 50
) -> list[Chunk]:
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")
    step = size - overlap
    chunks: list[Chunk] = []
    for doc in docs:
        text = " ".join(doc.text.split())
        if not text:
            continue
        start = 0
        index = 0
        while True:
            end = min(start + size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append(
                    Chunk(
                        id=f"{doc.id}#{index}",
                        doc_id=doc.id,
                        title=doc.title,
                        text=piece,
                    )
                )
                index += 1
            if end >= len(text):
                break
            start += step
    return chunks
