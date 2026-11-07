from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence

from ..retrieval.base import RetrievedChunk
from ..text import tokenize
from .base import Generation, build_prompt


class OllamaGenerator:
    """Local generation through the Ollama chat API."""

    def __init__(
        self,
        model: str,
        *,
        host: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
    ) -> None:
        self.name = model
        self._host = host.rstrip("/")
        self._timeout = timeout

    def generate(self, question: str, contexts: Sequence[RetrievedChunk]) -> Generation:
        payload = {
            "model": self.name,
            "messages": [{"role": "user", "content": build_prompt(question, contexts)}],
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self._host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"Ollama is not reachable at {self._host}: {error.reason}"
            ) from error
        text = (data.get("message") or {}).get("content", "").strip() or "I do not know."
        elapsed = int((time.perf_counter() - started) * 1000)
        tokens = int(data.get("eval_count") or len(tokenize(text)))
        return Generation(text=text, latency_ms=elapsed, tokens=tokens)
