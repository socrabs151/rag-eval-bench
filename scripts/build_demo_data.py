from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_eval_bench.evaluation.metrics import (  # noqa: E402
    recall_at_k,
    reciprocal_rank,
    token_f1,
)

DATA = ROOT / "data"
DATASET = "ml-glossary"

CORPUS = [
    {
        "id": "rag",
        "title": "Retrieval-augmented generation",
        "text": "Retrieval-augmented generation combines a retriever and a generator. "
        "The retriever selects passages from an external corpus, and the generator "
        "conditions its answer on those passages.",
    },
    {
        "id": "bm25",
        "title": "BM25",
        "text": "BM25 is a lexical ranking function built on term frequency and inverse "
        "document frequency. It uses the parameters k1 and b to control term saturation "
        "and document length normalization.",
    },
    {
        "id": "dense",
        "title": "Dense retrieval",
        "text": "Dense retrieval maps queries and passages into a shared vector space with "
        "a bi-encoder. Similarity search then runs over the encoded vectors, often with an "
        "approximate nearest neighbour index.",
    },
    {
        "id": "hybrid",
        "title": "Hybrid retrieval",
        "text": "Hybrid retrieval combines lexical and dense results. Reciprocal rank fusion "
        "is a common fusion method: each document scores as the sum of one over k plus its "
        "rank across the input rankings.",
    },
    {
        "id": "reranking",
        "title": "Reranking",
        "text": "A reranker is usually a cross-encoder: it scores the query and the passage "
        "jointly, so it cannot precompute passage vectors. Reranking the top candidates is "
        "more accurate and slower than the first retrieval stage.",
    },
    {
        "id": "recall",
        "title": "Recall at k",
        "text": "Recall at k measures the fraction of queries for which a relevant passage "
        "appears in the top k retrieved results. It ignores the exact rank inside the cut.",
    },
    {
        "id": "mrr",
        "title": "Mean reciprocal rank",
        "text": "Mean reciprocal rank averages the reciprocal rank of the first relevant "
        "result across queries. A relevant result at rank one contributes one, at rank two "
        "one half, and a missing result contributes zero.",
    },
    {
        "id": "faithfulness",
        "title": "Faithfulness",
        "text": "Faithfulness is the degree to which every claim in an answer is supported "
        "by the retrieved context. An answer can be correct and still unfaithful when it "
        "uses knowledge that the context does not contain.",
    },
    {
        "id": "judge",
        "title": "LLM-as-a-judge",
        "text": "An LLM judge scores model outputs against a rubric. A judge should be "
        "calibrated against human labels, for example with Cohen's kappa, before its scores "
        "are treated as evidence.",
    },
    {
        "id": "chunking",
        "title": "Chunking",
        "text": "Chunking splits documents into windows before indexing. Chunk overlap trades "
        "context completeness against retrieval precision: more overlap keeps sentences "
        "whole and adds near-duplicate passages.",
    },
    {
        "id": "qa-metrics",
        "title": "Exact match and token F1",
        "text": "Exact match and token F1 compare a predicted answer with the gold answer "
        "after normalization. Token F1 counts the overlap of unigrams and balances precision "
        "against recall.",
    },
    {
        "id": "contamination",
        "title": "Evaluation contamination",
        "text": "Contamination happens when evaluation questions or their answers leak into "
        "training data. It inflates scores and hides the difference between memorization and "
        "generalization.",
    },
]

QUESTIONS = [
    {
        "id": "q1",
        "question": "What does retrieval-augmented generation combine?",
        "expected_answer": "A retriever and a generator.",
        "dataset": DATASET,
        "gold_doc_ids": ["rag"],
        "tags": ["architecture"],
    },
    {
        "id": "q2",
        "question": "Which parameters does BM25 use?",
        "expected_answer": "k1 and b.",
        "dataset": DATASET,
        "gold_doc_ids": ["bm25"],
        "tags": ["retrieval"],
    },
    {
        "id": "q3",
        "question": "Which fusion method is common for hybrid retrieval?",
        "expected_answer": "Reciprocal rank fusion.",
        "dataset": DATASET,
        "gold_doc_ids": ["hybrid"],
        "tags": ["retrieval"],
    },
    {
        "id": "q4",
        "question": "Why is a cross-encoder slower than a bi-encoder?",
        "expected_answer": "It scores the query and the passage jointly.",
        "dataset": DATASET,
        "gold_doc_ids": ["reranking"],
        "tags": ["reranking"],
    },
    {
        "id": "q5",
        "question": "What does recall at k measure?",
        "expected_answer": "The fraction of queries with a relevant passage in the top k.",
        "dataset": DATASET,
        "gold_doc_ids": ["recall"],
        "tags": ["metrics"],
    },
    {
        "id": "q6",
        "question": "How is faithfulness defined?",
        "expected_answer": "Every claim in the answer is supported by the retrieved context.",
        "dataset": DATASET,
        "gold_doc_ids": ["faithfulness"],
        "tags": ["metrics"],
    },
    {
        "id": "q7",
        "question": "What should an LLM judge be calibrated against?",
        "expected_answer": "Human labels, for example with Cohen's kappa.",
        "dataset": DATASET,
        "gold_doc_ids": ["judge"],
        "tags": ["evaluation"],
    },
    {
        "id": "q8",
        "question": "What does chunk overlap trade off?",
        "expected_answer": "Context completeness against retrieval precision.",
        "dataset": DATASET,
        "gold_doc_ids": ["chunking"],
        "tags": ["indexing"],
    },
]

RUNS = [
    {
        "id": "exp-001",
        "name": "bm25-k5-extractive",
        "tags": ["baseline", "lexical"],
        "git_sha": "d4f1c2a",
        "seed": 13,
        "started_at": "2026-11-02T09:12:00Z",
        "index_seconds": 15,
        "duration_seconds": 148,
        "config": {
            "retriever": "bm25",
            "k": 5,
            "reranker": None,
            "generator": "qwen2.5:3b-instruct",
            "prompt": "v1",
            "chunk_size": 400,
            "chunk_overlap": 50,
        },
        "faithfulness": 0.71,
        "retrieved": {
            "q1": ["rag"],
            "q2": ["bm25"],
            "q3": ["dense", "rag"],
            "q4": ["dense", "reranking"],
            "q5": ["mrr", "retrieval"],
            "q6": ["faithfulness"],
            "q7": ["judge"],
            "q8": ["dense", "chunking"],
        },
        "answers": {
            "q1": "A retriever and a generator.",
            "q2": "BM25 uses the parameters k1 and b.",
            "q3": "Hybrid retrieval combines lexical and dense results.",
            "q4": "It is more accurate but slower.",
            "q5": (
                "Recall at k measures the fraction of queries with a relevant passage in the top k."
            ),
            "q6": (
                "Faithfulness means every claim in the answer is supported by the "
                "retrieved context."
            ),
            "q7": "An LLM judge should be calibrated against human labels.",
            "q8": "Chunk overlap trades context completeness against retrieval precision.",
        },
        "latencies": [1700, 1800, 1750, 1900, 1850, 1950, 1850, 1900],
    },
    {
        "id": "exp-002",
        "name": "hybrid-rerank-qwen3b",
        "tags": ["hybrid", "reranker"],
        "git_sha": "8a7e310",
        "seed": 13,
        "started_at": "2026-11-03T14:05:00Z",
        "index_seconds": 20,
        "duration_seconds": 212,
        "config": {
            "retriever": "hybrid",
            "k": 5,
            "reranker": "bge-reranker-base",
            "generator": "qwen2.5:3b-instruct",
            "prompt": "v1",
            "chunk_size": 400,
            "chunk_overlap": 50,
        },
        "faithfulness": 0.83,
        "retrieved": {
            "q1": ["rag"],
            "q2": ["bm25"],
            "q3": ["hybrid", "dense"],
            "q4": ["dense", "reranking"],
            "q5": ["recall", "mrr"],
            "q6": ["faithfulness"],
            "q7": ["prompting", "judge"],
            "q8": ["chunking", "dense"],
        },
        "answers": {
            "q1": "A retriever and a generator.",
            "q2": "BM25 uses k1 and b.",
            "q3": "Reciprocal rank fusion.",
            "q4": "It scores the query and the passage jointly, so it is slower.",
            "q5": "The fraction of queries with a relevant passage in the top k.",
            "q6": "Every claim in the answer is supported by the retrieved context.",
            "q7": "Human labels, for example with Cohen's kappa.",
            "q8": "Context completeness against retrieval precision.",
        },
        "latencies": [2400, 2500, 2600, 2700, 2550, 2650, 2600, 2500],
    },
    {
        "id": "exp-003",
        "name": "hybrid-rerank-qwen7b",
        "tags": ["hybrid", "reranker", "large-generator"],
        "git_sha": "f93c0b5",
        "seed": 13,
        "started_at": "2026-11-04T18:40:00Z",
        "index_seconds": 20,
        "duration_seconds": 305,
        "config": {
            "retriever": "hybrid",
            "k": 5,
            "reranker": "bge-reranker-base",
            "generator": "qwen2.5:7b-instruct",
            "prompt": "v1",
            "chunk_size": 400,
            "chunk_overlap": 50,
        },
        "faithfulness": 0.88,
        "retrieved": {
            "q1": ["rag"],
            "q2": ["bm25"],
            "q3": ["hybrid", "dense"],
            "q4": ["dense", "reranking"],
            "q5": ["recall", "mrr"],
            "q6": ["faithfulness"],
            "q7": ["prompting", "judge"],
            "q8": ["chunking", "dense"],
        },
        "answers": {
            "q1": "A retriever and a generator.",
            "q2": "k1 and b.",
            "q3": "Reciprocal rank fusion.",
            "q4": "It scores the query and the passage together.",
            "q5": "The fraction of queries with a relevant passage in the top k.",
            "q6": "Every claim in the answer is supported by the retrieved context.",
            "q7": "Human labels, for example with Cohen's kappa.",
            "q8": "Context completeness versus retrieval precision.",
        },
        "latencies": [3900, 4000, 4100, 4200, 4050, 4150, 4200, 4100],
    },
]


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _cumulative(values: list[float]) -> list[float]:
    return [round(mean(values[: index + 1]), 3) for index in range(len(values))]


def build_run(spec: dict) -> dict:
    started = _parse(spec["started_at"])
    prepare_end = started + timedelta(seconds=3)
    index_end = started + timedelta(seconds=spec["index_seconds"])
    finished = started + timedelta(seconds=spec["duration_seconds"])

    examples = []
    f1_values: list[float] = []
    recall_values: list[float] = []
    rr_values: list[float] = []
    for question, latency in zip(QUESTIONS, spec["latencies"], strict=True):
        answer = spec["answers"][question["id"]]
        retrieved = spec["retrieved"][question["id"]]
        f1 = token_f1(answer, question["expected_answer"])
        f1_values.append(f1)
        recall_values.append(recall_at_k(retrieved, question["gold_doc_ids"], 5))
        rr_values.append(reciprocal_rank(retrieved, question["gold_doc_ids"]))
        examples.append(
            {
                "id": question["id"],
                "question": question["question"],
                "expected": question["expected_answer"],
                "answer": answer,
                "retrieved_ids": retrieved,
                "f1": round(f1, 3),
                "faithfulness": None,
                "latency_ms": latency,
            }
        )

    return {
        "id": spec["id"],
        "name": spec["name"],
        "status": "succeeded",
        "dataset": DATASET,
        "owner": "socrabs151",
        "git_sha": spec["git_sha"],
        "seed": spec["seed"],
        "tags": spec["tags"],
        "started_at": _iso(started),
        "finished_at": _iso(finished),
        "duration_ms": spec["duration_seconds"] * 1000,
        "error": None,
        "config": spec["config"],
        "metrics": {
            "recall_at_5": round(mean(recall_values), 3),
            "mrr_at_10": round(mean(rr_values), 3),
            "f1": round(mean(f1_values), 3),
            "faithfulness": spec["faithfulness"],
            "latency_ms": round(mean(spec["latencies"])),
        },
        "series": {
            "f1": [
                {"step": step, "value": value} for step, value in enumerate(_cumulative(f1_values))
            ],
            "recall_at_5": [
                {"step": step, "value": value}
                for step, value in enumerate(_cumulative(recall_values))
            ],
            "mrr_at_10": [
                {"step": step, "value": value} for step, value in enumerate(_cumulative(rr_values))
            ],
            "latency_ms": [
                {"step": step, "value": round(value, 0)}
                for step, value in enumerate(_cumulative(spec["latencies"]))
            ],
        },
        "stages": [
            {
                "name": "prepare",
                "status": "succeeded",
                "started_at": _iso(started),
                "ended_at": _iso(prepare_end),
            },
            {
                "name": "index",
                "status": "succeeded",
                "started_at": _iso(prepare_end),
                "ended_at": _iso(index_end),
            },
            {
                "name": "evaluate",
                "status": "succeeded",
                "started_at": _iso(index_end),
                "ended_at": _iso(finished),
            },
        ],
        "examples": examples,
    }


def main() -> None:
    (DATA / "runs").mkdir(parents=True, exist_ok=True)

    with (DATA / "corpus.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for doc in CORPUS:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")

    with (DATA / "questions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for question in QUESTIONS:
            handle.write(json.dumps(question, ensure_ascii=False) + "\n")

    for spec in RUNS:
        record = build_run(spec)
        path = DATA / "runs" / f"{record['id']}.json"
        path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    evals = [
        {
            "id": "qa-f1",
            "name": "Answer token F1",
            "dataset": DATASET,
            "scorer": "token-f1",
            "metric": "f1",
            "metric_key": "f1",
            "direction": "maximize",
            "aggregate": "mean",
        }
    ]
    (DATA / "evals.json").write_text(
        json.dumps(evals, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    datasets = [
        {
            "id": DATASET,
            "name": "ML glossary QA",
            "version": "v1",
            "created_at": "2026-10-28T10:00:00Z",
            "tags": ["qa", "rag", "english"],
            "datasheet": [
                {
                    "label": "Motivation",
                    "value": "A small offline fixture for developing the evaluation harness "
                    "before a real corpus is connected.",
                    "mono": False,
                },
                {
                    "label": "Composition",
                    "value": "Twelve glossary notes and eight questions about retrieval, "
                    "generation and evaluation.",
                    "mono": False,
                },
                {
                    "label": "Collection",
                    "value": "The notes were written for this repository; no third-party "
                    "text is included.",
                    "mono": False,
                },
                {
                    "label": "Intended use",
                    "value": "Smoke tests, UI development and metric sanity checks. Not a "
                    "benchmark: the sample is too small for comparative claims.",
                    "mono": False,
                },
            ],
            "provenance": [
                {
                    "label": "Source",
                    "value": "scripts/build_demo_data.py",
                    "mono": True,
                },
                {
                    "label": "License",
                    "value": "Same as this repository",
                    "mono": False,
                },
            ],
        }
    ]
    (DATA / "datasets.json").write_text(
        json.dumps(datasets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for spec in RUNS:
        record = build_run(spec)
        print(
            f"{record['id']}: f1={record['metrics']['f1']:.3f} "
            f"recall@5={record['metrics']['recall_at_5']:.3f} "
            f"mrr@10={record['metrics']['mrr_at_10']:.3f}"
        )


if __name__ == "__main__":
    main()
