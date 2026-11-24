from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from rag_eval_bench import datasets
from rag_eval_bench.datasets import (
    Artifact,
    ChecksumError,
    DatasetError,
    DatasetSpec,
    convert_scifact,
    ensure_artifacts,
    fetch_dataset,
    read_manifest,
)
from rag_eval_bench.evaluation.runner import run_experiment
from rag_eval_bench.storage import Store

FIXTURE = Path(__file__).parent / "fixtures" / "scifact_mini"
SOURCE_FILES = ("_source/corpus.jsonl", "_source/queries.jsonl", "_source/qrels/test.tsv")


class FakeOpener:
    def __init__(self, files: dict[str, bytes]) -> None:
        self._files = files
        self.calls: list[str] = []

    def __call__(self, url: str, timeout: int = 60) -> BytesIO:  # noqa: ARG002
        self.calls.append(url)
        return BytesIO(self._files[url])


def _fixture_spec() -> DatasetSpec:
    artifacts = []
    for relative in SOURCE_FILES:
        path = FIXTURE / relative
        artifacts.append(
            Artifact(
                path=relative,
                url=f"test://{relative}",
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size=path.stat().st_size,
            )
        )
    return DatasetSpec(
        id="scifact",
        title="SciFact fixture",
        version="fixture",
        source="tests/fixtures/scifact_mini",
        license="test only",
        artifacts=tuple(artifacts),
        datasheet=(("Intended use", "Offline tests."),),
    )


@pytest.fixture()
def fixture_files() -> dict[str, bytes]:
    return {f"test://{relative}": (FIXTURE / relative).read_bytes() for relative in SOURCE_FILES}


@pytest.fixture()
def fixture_spec(monkeypatch: pytest.MonkeyPatch) -> DatasetSpec:
    spec = _fixture_spec()
    monkeypatch.setitem(datasets.SPECS, "scifact", spec)
    return spec


def test_convert_writes_store_layout(tmp_path: Path) -> None:
    record, documents, questions = convert_scifact(FIXTURE, tmp_path)

    assert (documents, questions) == (3, 2)
    assert record.id == "scifact"

    store = Store(tmp_path)
    assert [doc.id for doc in store.corpus] == ["101", "102", "103"]
    assert [item.id for item in store.questions] == ["1", "2"]
    assert store.questions[1].gold_doc_ids == ["102", "103"]
    assert all(item.expected_answer == "" for item in store.questions)
    assert store.datasets[0].id == "scifact"


def test_fetch_downloads_verified_files_and_writes_manifest(
    tmp_path: Path, fixture_spec: DatasetSpec, fixture_files: dict[str, bytes]
) -> None:
    opener = FakeOpener(fixture_files)

    _, documents, questions = fetch_dataset("scifact", tmp_path, opener=opener)

    assert len(opener.calls) == len(fixture_spec.artifacts)
    assert (documents, questions) == (3, 2)
    for artifact in fixture_spec.artifacts:
        assert (tmp_path / artifact.path).is_file()

    manifest = read_manifest(tmp_path)
    assert manifest["documents"] == 3
    assert manifest["questions"] == 2
    assert {item["sha256"] for item in manifest["files"]} == {
        artifact.sha256 for artifact in fixture_spec.artifacts
    }


def test_fetch_reuses_valid_files(
    tmp_path: Path, fixture_spec: DatasetSpec, fixture_files: dict[str, bytes]
) -> None:
    opener = FakeOpener(fixture_files)
    fetch_dataset("scifact", tmp_path, opener=opener)

    opener.calls.clear()
    fetch_dataset("scifact", tmp_path, opener=opener)

    assert opener.calls == []


def test_checksum_mismatch_is_reported(
    tmp_path: Path, fixture_spec: DatasetSpec, fixture_files: dict[str, bytes]
) -> None:
    broken = replace(
        fixture_spec,
        artifacts=(
            replace(fixture_spec.artifacts[0], sha256="0" * 64),
            *fixture_spec.artifacts[1:],
        ),
    )
    opener = FakeOpener(fixture_files)

    with pytest.raises(ChecksumError):
        ensure_artifacts(broken, tmp_path, opener=opener)


def test_unknown_dataset_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetError):
        fetch_dataset("does-not-exist", tmp_path)


def test_retrieval_only_dataset_has_no_answer_metrics(tmp_path: Path) -> None:
    convert_scifact(FIXTURE, tmp_path)
    store = Store(tmp_path)

    record = run_experiment(store=store, artifacts_dir=tmp_path / "artifacts", k=3)

    assert record.dataset == "scifact"
    assert "f1" not in record.metrics
    assert "f1" not in record.series
    assert record.metrics["recall_at_5"] >= 0.0
    assert record.metrics["mrr_at_10"] >= 0.0


def test_datasets_file_is_valid_json(tmp_path: Path) -> None:
    convert_scifact(FIXTURE, tmp_path)

    raw: list[dict[str, Any]] = json.loads((tmp_path / "datasets.json").read_text(encoding="utf-8"))

    assert raw[0]["id"] == "scifact"
    assert raw[0]["provenance"]
