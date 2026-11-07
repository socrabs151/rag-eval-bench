from __future__ import annotations

from statistics import mean

import frontedge
from fastapi import Request

from ..storage import Store


class EvalsProvider:
    def __init__(self, store: Store) -> None:
        self._store = store

    async def list_evals(self, request: Request) -> list[frontedge.EvalSummary]:
        runs = [record for record in self._store.runs() if record.examples]
        updated_at = max((record.started_at for record in runs), default=None)
        return [
            frontedge.EvalSummary(
                id=definition.id,
                name=definition.name,
                dataset=definition.dataset,
                scorer=definition.scorer,
                metric=definition.metric,
                direction=frontedge.MetricDirection(definition.direction),
                aggregate=definition.aggregate,
                runs=len(runs),
                updated_at=updated_at,
            )
            for definition in self._store.evals
        ]

    async def compare_evals(
        self,
        request: Request,
        eval_id: str,
        reference_run: str,
        comparison_run: str,
    ) -> frontedge.EvalComparison | None:
        definition = self._store.eval(eval_id)
        reference = self._store.run(reference_run)
        comparison = self._store.run(comparison_run)
        if definition is None or reference is None or comparison is None:
            return None
        if not reference.examples or not comparison.examples:
            return None

        comparison_rows = {row.id: row for row in comparison.examples}
        examples: list[frontedge.ExampleResult] = []
        for row in reference.examples:
            other = comparison_rows.get(row.id)
            if other is None:
                continue
            examples.append(
                frontedge.ExampleResult(
                    id=row.id,
                    input=row.question,
                    output=other.answer,
                    expected=row.expected,
                    reference_output=row.answer,
                    comparison_output=other.answer,
                    reference_score=getattr(row, definition.metric_key),
                    comparison_score=getattr(other, definition.metric_key),
                    changed=row.answer != other.answer,
                )
            )

        return frontedge.EvalComparison(
            eval_id=definition.id,
            name=definition.name,
            dataset=definition.dataset,
            scorer=definition.scorer,
            metric=definition.metric,
            direction=frontedge.MetricDirection(definition.direction),
            aggregate=definition.aggregate,
            reference_run=reference.id,
            comparison_run=comparison.id,
            reference_name=reference.name,
            comparison_name=comparison.name,
            examples=examples,
            reference_aggregate=round(
                mean(getattr(row, definition.metric_key) for row in reference.examples), 3
            ),
            comparison_aggregate=round(
                mean(getattr(row, definition.metric_key) for row in comparison.examples), 3
            ),
            changed=sum(1 for example in examples if example.changed),
        )
