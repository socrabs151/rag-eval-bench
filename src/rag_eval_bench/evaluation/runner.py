from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from ..chunking import Chunk, chunk_documents
from ..generation.base import Generator
from ..generation.mock import ExtractiveGenerator
from ..generation.ollama import OllamaGenerator
from ..retrieval.base import Retriever
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.dense import DenseRetriever
from ..retrieval.hybrid import HybridRetriever
from ..schemas import ExampleRecord, ExperimentConfig, RunRecord, SeriesPoint, StageRecord
from ..storage import Store
from .metrics import recall_at_k, reciprocal_rank, token_f1


def build_retriever(name: str, chunks: Sequence[Chunk]) -> Retriever:
    if name == "bm25":
        return BM25Retriever(chunks)
    if name == "dense":
        return DenseRetriever(chunks)
    if name == "hybrid":
        return HybridRetriever([BM25Retriever(chunks), DenseRetriever(chunks)])
    raise ValueError(f"unknown retriever: {name}")


def build_generator(name: str, model: str) -> Generator:
    if name == "extractive":
        return ExtractiveGenerator()
    if name == "ollama":
        return OllamaGenerator(model)
    raise ValueError(f"unknown generator: {name}")


def run_experiment(
    *,
    store: Store,
    artifacts_dir: Path | str,
    retriever: str = "bm25",
    k: int = 5,
    generator: str = "extractive",
    model: str = "qwen2.5:3b-instruct",
    name: str | None = None,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> RunRecord:
    if not store.corpus:
        raise RuntimeError("no corpus documents found")
    if not store.questions:
        raise RuntimeError("no questions found")

    started = datetime.now(UTC)
    chunks = chunk_documents(store.corpus, size=chunk_size, overlap=chunk_overlap)
    retriever_impl = build_retriever(retriever, chunks)
    generator_impl = build_generator(generator, model)
    indexed_at = datetime.now(UTC)

    examples: list[ExampleRecord] = []
    f1_values: list[float] = []
    recall_values: list[float] = []
    rr_values: list[float] = []
    latencies: list[int] = []
    series_f1: list[SeriesPoint] = []
    series_recall: list[SeriesPoint] = []
    series_mrr: list[SeriesPoint] = []
    series_latency: list[SeriesPoint] = []

    for step, question in enumerate(store.questions):
        tick = time.perf_counter()
        hits = retriever_impl.search(question.question, k)
        retrieved_ids = list(dict.fromkeys(hit.chunk.doc_id for hit in hits))
        generation = generator_impl.generate(question.question, hits)
        latency = int((time.perf_counter() - tick) * 1000)

        f1 = token_f1(generation.text, question.expected_answer)
        recall = recall_at_k(retrieved_ids, question.gold_doc_ids, k)
        rr = reciprocal_rank(retrieved_ids, question.gold_doc_ids)

        f1_values.append(f1)
        recall_values.append(recall)
        rr_values.append(rr)
        latencies.append(latency)
        examples.append(
            ExampleRecord(
                id=question.id,
                question=question.question,
                expected=question.expected_answer,
                answer=generation.text,
                retrieved_ids=retrieved_ids,
                f1=round(f1, 3),
                faithfulness=None,
                latency_ms=latency,
            )
        )
        series_f1.append(SeriesPoint(step=step, value=round(mean(f1_values), 3)))
        series_recall.append(SeriesPoint(step=step, value=round(mean(recall_values), 3)))
        series_mrr.append(SeriesPoint(step=step, value=round(mean(rr_values), 3)))
        series_latency.append(SeriesPoint(step=step, value=round(mean(latencies), 0)))

    finished = datetime.now(UTC)
    run_id = _unique_run_id(Path(artifacts_dir), retriever, started)
    record = RunRecord(
        id=run_id,
        name=name or f"{retriever}-k{k}-{generator}",
        status="succeeded",
        dataset="ml-glossary",
        tags=[retriever, generator],
        started_at=started,
        finished_at=finished,
        duration_ms=int((finished - started).total_seconds() * 1000),
        config=ExperimentConfig(
            retriever=retriever,
            k=k,
            generator=generator if generator == "extractive" else model,
            prompt="v1",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
        metrics={
            "recall_at_5": round(mean(recall_values), 3),
            "mrr_at_10": round(mean(rr_values), 3),
            "f1": round(mean(f1_values), 3),
            "faithfulness": 0.0,
            "latency_ms": round(mean(latencies)),
        },
        series={
            "f1": series_f1,
            "recall_at_5": series_recall,
            "mrr_at_10": series_mrr,
            "latency_ms": series_latency,
        },
        stages=[
            StageRecord(name="prepare", status="succeeded", started_at=started),
            StageRecord(
                name="index",
                status="succeeded",
                started_at=started,
                ended_at=indexed_at,
            ),
            StageRecord(
                name="evaluate",
                status="succeeded",
                started_at=indexed_at,
                ended_at=finished,
            ),
        ],
        examples=examples,
    )

    runs_dir = Path(artifacts_dir) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{run_id}.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return record


def _unique_run_id(artifacts_dir: Path, retriever: str, started: datetime) -> str:
    base = f"{retriever}-{started:%Y%m%d-%H%M%S}"
    candidate = base
    counter = 2
    while (artifacts_dir / "runs" / f"{candidate}.json").exists():
        candidate = f"{base}-{counter}"
        counter += 1
    return candidate
