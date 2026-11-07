from pathlib import Path

from rag_eval_bench.storage import Store

ROOT = Path(__file__).resolve().parents[1]


def test_store_loads_the_demo_data() -> None:
    store = Store(ROOT / "data")
    assert len(store.corpus) == 12
    assert len(store.questions) == 8
    assert len(store.datasets) == 1
    assert len(store.evals) == 1
    assert len(store.runs()) == 3


def test_store_returns_a_run_by_id() -> None:
    store = Store(ROOT / "data")
    run = store.run("exp-001")
    assert run is not None
    assert run.metrics["recall_at_5"] == 0.75
    assert len(run.examples) == 8


def test_store_returns_none_for_unknown_ids() -> None:
    store = Store(ROOT / "data")
    assert store.run("missing") is None
    assert store.dataset("missing") is None
    assert store.eval("missing") is None
