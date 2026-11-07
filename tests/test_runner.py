from pathlib import Path

from rag_eval_bench.evaluation.runner import run_experiment
from rag_eval_bench.storage import Store

ROOT = Path(__file__).resolve().parents[1]


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
