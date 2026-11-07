from __future__ import annotations

import argparse
from collections.abc import Sequence

from .config import Settings
from .evaluation.runner import run_experiment
from .storage import Store


def main(argv: Sequence[str] | None = None) -> int:
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

    serve_parser = subparsers.add_parser("serve", help="serve the interface")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")

    args = parser.parse_args(argv)
    settings = Settings.from_env()

    if args.command == "serve":
        import uvicorn

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
    )
    print(
        f"{record.id}: f1={record.metrics['f1']:.3f} "
        f"recall@5={record.metrics['recall_at_5']:.3f} "
        f"mrr@10={record.metrics['mrr_at_10']:.3f} "
        f"latency={record.metrics['latency_ms']:.0f} ms"
    )
    return 0
