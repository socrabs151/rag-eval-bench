from __future__ import annotations

import frontedge
from fastapi import Request
from frontedge.models import ExamplesPage, ExamplesQuery, RunsPage, RunsQuery

from ..evaluation.metrics import METRIC_DIRECTIONS
from ..schemas import RunRecord
from ..storage import Store


def _to_frontedge(record: RunRecord) -> frontedge.Run:
    return frontedge.Run(
        id=record.id,
        name=record.name,
        status=frontedge.RunStatus(record.status),
        dataset=record.dataset,
        git_sha=record.git_sha,
        started_at=record.started_at,
        finished_at=record.finished_at,
        duration_ms=record.duration_ms,
        owner=record.owner,
        tags=record.tags,
        seed=record.seed,
        params=record.config.model_dump(exclude_none=True),
        error=record.error,
        metrics=record.metrics,
        series={
            metric: [frontedge.MetricPoint(step=point.step, value=point.value) for point in points]
            for metric, points in record.series.items()
        },
        metric_direction={
            metric: frontedge.MetricDirection(direction)
            for metric, direction in METRIC_DIRECTIONS.items()
        },
        stages=[
            frontedge.Stage(
                name=stage.name,
                status=frontedge.RunStatus(stage.status),
                started_at=stage.started_at,
                ended_at=stage.ended_at,
            )
            for stage in record.stages
        ],
    )


def _matches(record: RunRecord, query: RunsQuery) -> bool:
    if query.status and record.status not in query.status:
        return False
    if query.dataset and record.dataset != query.dataset:
        return False
    if query.tag and query.tag not in record.tags:
        return False
    if query.search:
        needle = query.search.lower()
        haystack = " ".join([record.id, record.name, record.dataset, *record.tags]).lower()
        if needle not in haystack:
            return False
    return True


def _sort_value(record: RunRecord, key: str | None) -> object:
    if key in (None, "", "startedAt", "started_at"):
        return record.started_at
    if key in ("durationMs", "duration_ms"):
        return record.duration_ms
    if key in ("name", "status", "dataset"):
        return getattr(record, key)
    return record.metrics.get(key)


def _sorted(records: list[RunRecord], key: str | None, order: str) -> list[RunRecord]:
    if key in (None, "", "startedAt", "started_at"):
        return sorted(records, key=lambda record: record.started_at, reverse=order != "asc")
    present = [record for record in records if _sort_value(record, key) is not None]
    missing = [record for record in records if _sort_value(record, key) is None]
    present.sort(
        key=lambda record: _sort_value(record, key),
        reverse=order == "desc",
    )
    return present + missing


def _page_bounds(offset: int, limit: int | None, total: int) -> tuple[int, int | None]:
    start = max(offset, 0)
    end = None if limit is None else start + limit
    return start, end


class RunsProvider:
    def __init__(self, store: Store) -> None:
        self._store = store

    async def list_runs(self, request: Request, query: RunsQuery) -> RunsPage:
        records = [record for record in self._store.runs() if _matches(record, query)]
        records = _sorted(records, query.sort, query.order)
        total = len(records)
        start, end = _page_bounds(query.offset, query.limit, total)
        return RunsPage(
            items=[_to_frontedge(record) for record in records[start:end]],
            total=total,
        )

    async def get_run(self, request: Request, run_id: str) -> frontedge.Run | None:
        record = self._store.run(run_id)
        return None if record is None else _to_frontedge(record)

    async def list_examples(
        self, request: Request, run_id: str, query: ExamplesQuery
    ) -> ExamplesPage:
        record = self._store.run(run_id)
        if record is None:
            return ExamplesPage(items=[], total=0)
        rows = []
        for example in record.examples:
            if query.search:
                needle = query.search.lower()
                haystack = " ".join([example.question, example.answer, example.expected]).lower()
                if needle not in haystack:
                    continue
            rows.append(
                frontedge.Example(
                    id=example.id,
                    input=example.question,
                    output=example.answer,
                    expected=example.expected,
                    score=example.f1,
                )
            )
        total = len(rows)
        start, end = _page_bounds(query.offset, query.limit, total)
        return ExamplesPage(items=rows[start:end], total=total)
