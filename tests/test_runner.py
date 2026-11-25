from pathlib import Path

import pytest

from rag_eval_bench import evaluation
from rag_eval_bench.evaluation.runner import build_reranker, run_experiment
from rag_eval_bench.retrieval.base import RetrievedChunk
from rag_eval_bench.storage import Store

ROOT = Path(__file__).resolve().parents[1]


class ReversingReranker:
    def rerank(self, query: str, hits: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        return list(reversed(hits))[:k]


def test_run_experiment_writes_a_loadable_run(tmp_path: Path) -> None:
    store = Store(ROOT / "data")
    record = run_experiment(
        store=store,
        artifacts_dir=tmp_path,
        retriever="bm25",
        k=3,
        generator="extractive",
    )

    assert record.status == "succeeded"
    assert len(record.examples) == 8
    assert 0.0 <= record.metrics["f1"] <= 1.0
    assert (tmp_path / "runs" / f"{record.id}.json").is_file()

    reloaded = Store(ROOT / "data", tmp_path)
    assert reloaded.run(record.id) is not None


def test_reranker_reorders_hits_and_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(ROOT / "data")
    monkeypatch.setattr(evaluation.runner, "build_reranker", lambda name: ReversingReranker())

    record = run_experiment(
        store=store,
        artifacts_dir=tmp_path,
        retriever="bm25",
        k=3,
        reranker="fake",
        candidates=10,
    )

    assert record.config.reranker == "fake"
    assert record.config.candidates == 10
    assert "fake" in record.tags
    assert "rerank_ms" in record.metrics
    assert "retrieval_ms" in record.metrics
    assert record.metrics["latency_ms"] >= record.metrics["retrieval_ms"]


def test_unknown_reranker_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown reranker"):
        build_reranker("does-not-exist")


def test_candidates_must_cover_k(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = Store(ROOT / "data")
    monkeypatch.setattr(evaluation.runner, "build_reranker", lambda name: ReversingReranker())

    with pytest.raises(ValueError, match="candidates"):
        run_experiment(
            store=store,
            artifacts_dir=tmp_path,
            retriever="bm25",
            k=5,
            reranker="fake",
            candidates=2,
        )
