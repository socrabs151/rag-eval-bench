from __future__ import annotations

import frontedge
from fastapi import Request
from frontedge.models import ExamplesQuery

from ..schemas import DatasetRecord
from ..storage import Store


class DatasetsProvider:
    def __init__(self, store: Store) -> None:
        self._store = store

    def _info(self, record: DatasetRecord) -> frontedge.DatasetInfo:
        examples_count = sum(
            1 for question in self._store.questions if question.dataset == record.id
        )
        return frontedge.DatasetInfo(
            id=record.id,
            name=record.name,
            version=record.version,
            examples_count=examples_count,
            created_at=record.created_at,
            tags=record.tags,
            datasheet=[
                frontedge.ProvenanceEntry(label=entry.label, value=entry.value, mono=entry.mono)
                for entry in record.datasheet
            ],
            provenance=[
                frontedge.ProvenanceEntry(label=entry.label, value=entry.value, mono=entry.mono)
                for entry in record.provenance
            ],
        )

    async def list_datasets(self, request: Request) -> list[frontedge.DatasetInfo]:
        return [self._info(record) for record in self._store.datasets]

    async def get_dataset(
        self, request: Request, dataset_id: str
    ) -> frontedge.DatasetDetail | None:
        record = self._store.dataset(dataset_id)
        if record is None:
            return None
        info = self._info(record)
        return frontedge.DatasetDetail(**info.model_dump(), examples=[])

    async def list_dataset_examples(
        self, request: Request, dataset_id: str, query: ExamplesQuery
    ) -> frontedge.DatasetExamplesPage:
        if self._store.dataset(dataset_id) is None:
            return frontedge.DatasetExamplesPage(items=[], total=0)
        corpus = {doc.id: doc for doc in self._store.corpus}
        rows: list[frontedge.ExampleResult] = []
        for question in self._store.questions:
            if question.dataset != dataset_id:
                continue
            gold = " ".join(
                corpus[doc_id].text for doc_id in question.gold_doc_ids if doc_id in corpus
            )
            if query.search:
                needle = query.search.lower()
                haystack = " ".join([question.question, gold, question.expected_answer]).lower()
                if needle not in haystack:
                    continue
            rows.append(
                frontedge.ExampleResult(
                    id=question.id,
                    input=question.question,
                    output=gold,
                    expected=question.expected_answer,
                    changed=False,
                )
            )
        total = len(rows)
        start = max(query.offset, 0)
        end = None if query.limit is None else start + query.limit
        return frontedge.DatasetExamplesPage(items=rows[start:end], total=total)
