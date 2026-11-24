from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval_bench import cli
from rag_eval_bench.datasets import DatasetError
from rag_eval_bench.schemas import DatasetRecord


def test_fetch_data_wires_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: dict[str, object] = {}

    def fake_fetch(name: str, destination: Path, *, force: bool = False) -> tuple:
        calls.update(name=name, destination=Path(destination), force=force)
        return DatasetRecord(id=name, name="SciFact", version="v1"), 5183, 300

    monkeypatch.setattr(cli, "fetch_dataset", fake_fetch)

    code = cli.main(["fetch-data", "--dataset", "scifact", "--dest", str(tmp_path)])

    assert code == 0
    assert calls == {"name": "scifact", "destination": tmp_path, "force": False}
    output = capsys.readouterr().out
    assert "5183" in output
    assert "300" in output


def test_fetch_data_default_destination(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_fetch(name: str, destination: Path, *, force: bool = False) -> tuple:
        seen["destination"] = Path(destination)
        return DatasetRecord(id=name, name="SciFact", version="v1"), 1, 1

    monkeypatch.setattr(cli, "fetch_dataset", fake_fetch)

    assert cli.main(["fetch-data"]) == 0
    assert seen["destination"] == Path("datasets") / "scifact"


def test_fetch_data_reports_dataset_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing_fetch(*args: object, **kwargs: object) -> tuple:
        raise DatasetError("boom")

    monkeypatch.setattr(cli, "fetch_dataset", failing_fetch)

    assert cli.main(["fetch-data"]) == 2
    assert "boom" in capsys.readouterr().err
