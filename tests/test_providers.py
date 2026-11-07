import asyncio
from pathlib import Path

from frontedge.models import ExamplesQuery, RunsQuery

from rag_eval_bench.providers import DatasetsProvider, EvalsProvider, RunsProvider
from rag_eval_bench.storage import Store

ROOT = Path(__file__).resolve().parents[1]


def _store() -> Store:
    return Store(ROOT / "data")


def test_runs_provider_pages_and_reports_the_total() -> None:
    provider = RunsProvider(_store())
    page = asyncio.run(provider.list_runs(None, RunsQuery(limit=2, offset=0)))
    assert page.total == 3
    assert len(page.items) == 2


def test_runs_provider_filters_by_status_and_search() -> None:
    provider = RunsProvider(_store())
    page = asyncio.run(provider.list_runs(None, RunsQuery(status=["succeeded"], search="hybrid")))
    assert page.total == 2
    assert all("hybrid" in run.name for run in page.items)


def test_runs_provider_sorts_by_metric() -> None:
    provider = RunsProvider(_store())
    page = asyncio.run(provider.list_runs(None, RunsQuery(sort="f1", order="desc")))
    values = [run.metrics["f1"] for run in page.items]
    assert values == sorted(values, reverse=True)


def test_runs_provider_serves_examples_a_page_at_a_time() -> None:
    provider = RunsProvider(_store())
    page = asyncio.run(provider.list_examples(None, "exp-001", ExamplesQuery(limit=3, offset=0)))
    assert page.total == 8
    assert len(page.items) == 3


def test_evals_provider_lists_and_compares_runs() -> None:
    provider = EvalsProvider(_store())
    evals = asyncio.run(provider.list_evals(None))
    assert len(evals) == 1
    assert evals[0].runs == 3

    comparison = asyncio.run(provider.compare_evals(None, "qa-f1", "exp-001", "exp-002"))
    assert comparison is not None
    assert len(comparison.examples) == 8
    assert comparison.comparison_aggregate > comparison.reference_aggregate
    assert comparison.changed > 0


def test_evals_provider_returns_none_for_unknown_ids() -> None:
    provider = EvalsProvider(_store())
    assert asyncio.run(provider.compare_evals(None, "missing", "exp-001", "exp-002")) is None
    assert asyncio.run(provider.compare_evals(None, "qa-f1", "exp-001", "missing")) is None


def test_datasets_provider_serves_metadata_and_examples() -> None:
    provider = DatasetsProvider(_store())
    datasets = asyncio.run(provider.list_datasets(None))
    assert len(datasets) == 1
    assert datasets[0].examples_count == 8

    detail = asyncio.run(provider.get_dataset(None, "ml-glossary"))
    assert detail is not None
    assert detail.version == "v1"

    page = asyncio.run(
        provider.list_dataset_examples(None, "ml-glossary", ExamplesQuery(limit=4, offset=0))
    )
    assert page.total == 8
    assert len(page.items) == 4
