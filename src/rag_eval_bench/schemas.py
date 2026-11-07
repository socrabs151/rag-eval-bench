from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CorpusDoc(Schema):
    id: str
    title: str
    text: str


class Question(Schema):
    id: str
    question: str
    expected_answer: str
    dataset: str = "ml-glossary"
    gold_doc_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class ExperimentConfig(Schema):
    retriever: str
    k: int = 5
    reranker: str | None = None
    generator: str
    prompt: str = "v1"
    chunk_size: int = 400
    chunk_overlap: int = 50


class ExampleRecord(Schema):
    id: str
    question: str
    expected: str
    answer: str
    retrieved_ids: list[str] = Field(default_factory=list)
    f1: float
    faithfulness: float | None = None
    latency_ms: int


class SeriesPoint(Schema):
    step: int
    value: float


class StageRecord(Schema):
    name: str
    status: str
    started_at: datetime
    ended_at: datetime | None = None


class RunRecord(Schema):
    id: str
    name: str
    status: str
    dataset: str
    owner: str = ""
    git_sha: str = ""
    seed: int | None = None
    tags: list[str] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: int | None = None
    error: str | None = None
    config: ExperimentConfig
    metrics: dict[str, float] = Field(default_factory=dict)
    series: dict[str, list[SeriesPoint]] = Field(default_factory=dict)
    stages: list[StageRecord] = Field(default_factory=list)
    examples: list[ExampleRecord] = Field(default_factory=list)


class EvalDefinition(Schema):
    id: str
    name: str
    dataset: str
    scorer: str
    metric: str
    metric_key: str = "f1"
    direction: Literal["maximize", "minimize", "none"] = "maximize"
    aggregate: Literal["mean", "fraction"] = "mean"


class DatasheetEntry(Schema):
    label: str
    value: str
    mono: bool = False


class DatasetRecord(Schema):
    id: str
    name: str
    version: str
    created_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    datasheet: list[DatasheetEntry] = Field(default_factory=list)
    provenance: list[DatasheetEntry] = Field(default_factory=list)
