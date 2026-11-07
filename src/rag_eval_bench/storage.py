from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .schemas import CorpusDoc, DatasetRecord, EvalDefinition, Question, RunRecord

ModelT = TypeVar("ModelT", bound=BaseModel)


def _read_jsonl(path: Path, model: type[ModelT]) -> list[ModelT]:
    if not path.is_file():
        return []
    items: list[ModelT] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(model.model_validate_json(line))
    return items


def _read_json_list(path: Path, model: type[ModelT]) -> list[ModelT]:
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [model.model_validate(item) for item in raw]


class Store:
    """Read-only access to the demo corpus and to experiment runs stored as JSON."""

    def __init__(self, data_dir: Path | str, artifacts_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self._corpus = _read_jsonl(self.data_dir / "corpus.jsonl", CorpusDoc)
        self._questions = _read_jsonl(self.data_dir / "questions.jsonl", Question)
        self._evals = _read_json_list(self.data_dir / "evals.json", EvalDefinition)
        self._datasets = _read_json_list(self.data_dir / "datasets.json", DatasetRecord)
        self._runs: dict[str, RunRecord] = {}
        directories = [self.data_dir / "runs"]
        if self.artifacts_dir is not None:
            directories.append(self.artifacts_dir / "runs")
        for directory in directories:
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.json")):
                record = RunRecord.model_validate_json(path.read_text(encoding="utf-8"))
                self._runs[record.id] = record

    @property
    def corpus(self) -> list[CorpusDoc]:
        return list(self._corpus)

    @property
    def questions(self) -> list[Question]:
        return list(self._questions)

    @property
    def evals(self) -> list[EvalDefinition]:
        return list(self._evals)

    @property
    def datasets(self) -> list[DatasetRecord]:
        return list(self._datasets)

    def runs(self) -> list[RunRecord]:
        return sorted(self._runs.values(), key=lambda run: run.started_at, reverse=True)

    def run(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    def dataset(self, dataset_id: str) -> DatasetRecord | None:
        return next((item for item in self._datasets if item.id == dataset_id), None)

    def eval(self, eval_id: str) -> EvalDefinition | None:
        return next((item for item in self._evals if item.id == eval_id), None)
