from __future__ import annotations

import re
import time
from collections.abc import Sequence

from ..retrieval.base import RetrievedChunk
from ..text import tokenize
from .base import Generation

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class ExtractiveGenerator:
    """A deterministic baseline: returns the context sentence closest to the question."""

    name = "extractive-mock"

    def generate(self, question: str, contexts: Sequence[RetrievedChunk]) -> Generation:
        started = time.perf_counter()
        question_terms = set(tokenize(question))
        best = ""
        best_score = -1.0
        for hit in contexts:
            for sentence in SENTENCE_RE.split(hit.chunk.text):
                sentence = sentence.strip()
                if not sentence:
                    continue
                terms = set(tokenize(sentence))
                if not terms or not question_terms:
                    continue
                score = len(question_terms & terms) / len(question_terms)
                if score > best_score:
                    best_score = score
                    best = sentence
        text = best or "I do not know."
        elapsed = int((time.perf_counter() - started) * 1000)
        return Generation(text=text, latency_ms=elapsed, tokens=len(tokenize(text)))
