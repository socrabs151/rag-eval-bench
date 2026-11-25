from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from .config import Settings
from .datasets import SPECS, DatasetError, fetch_dataset
from .evaluation.runner import run_experiment
from .retrieval.rerank import RERANKER_MODELS
from .storage import Store


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-eval-bench",
        description="Run RAG experiments and serve the frontedge interface.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run one experiment on the corpus")
    run_parser.add_argument("--retriever", default="bm25", choices=["bm25", "dense", "hybrid"])
    run_parser.add_argument("--k", type=int, default=5)
    run_parser.add_argument("--generator", default="extractive", choices=["extractive", "ollama"])
    run_parser.add_argument("--model", default="qwen2.5:3b-instruct")
    run_parser.add_argument("--name")
    run_parser.add_argument(
        "--reranker",
        default="none",
        choices=["none", *sorted(RERANKER_MODELS)],
        help="cross-encoder reranker applied to the retrieved candidates",
    )
    run_parser.add_argument(
        "--candidates",
        type=int,
        default=20,
        help="candidates passed to the reranker (default: 20)",
    )
    run_parser.add_argument(
        "--dense-model",
        help="sentence-transformers model for dense retrieval",
    )
    run_parser.add_argument(
        "--data-dir",
        type=Path,
        help="directory with corpus.jsonl and questions.jsonl (default: data)",
    )

    serve_parser = subparsers.add_parser("serve", help="serve the interface")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    serve_parser.add_argument(
        "--data-dir",
        type=Path,
        help="directory with corpus.jsonl and questions.jsonl (default: data)",
    )

    fetch_parser = subparsers.add_parser(
        "fetch-data", help="download, verify and convert a public dataset"
    )
    fetch_parser.add_argument("--dataset", default="scifact", choices=sorted(SPECS))
    fetch_parser.add_argument(
        "--dest",
        type=Path,
        help="target directory (default: datasets/<dataset>)",
    )
    fetch_parser.add_argument(
        "--force",
        action="store_true",
        help="download again even when the local files are valid",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "fetch-data":
        destination = args.dest or Path("datasets") / args.dataset
        try:
            record, documents, questions = fetch_dataset(
                args.dataset, destination, force=args.force
            )
        except DatasetError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        print(
            f"{record.id} ({record.version}): {documents} documents, "
            f"{questions} questions -> {destination}"
        )
        return 0

    settings = Settings.from_env()
    if args.data_dir is not None:
        settings = replace(settings, data_dir=args.data_dir)

    if args.command == "serve":
        import uvicorn

        os.environ["RAG_EVAL_DATA_DIR"] = str(settings.data_dir)
        uvicorn.run(
            "rag_eval_bench.app:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
        return 0

    store = Store(settings.data_dir, settings.artifacts_dir)
    record = run_experiment(
        store=store,
        artifacts_dir=settings.artifacts_dir,
        retriever=args.retriever,
        k=args.k,
        generator=args.generator,
        model=args.model,
        name=args.name,
        reranker=args.reranker,
        candidates=args.candidates,
        dense_model=args.dense_model,
    )
    details = " ".join(f"{key}={value}" for key, value in record.metrics.items())
    print(f"{record.id}: {details}")
    return 0
