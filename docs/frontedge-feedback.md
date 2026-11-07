# Frontedge feedback log

Findings about the `frontedge` dependency (`0.5.x`) collected while building this bench. The
purpose is to keep each observation reproducible and ready to file upstream, not to keep a wish
list.

## Conventions

- One entry per finding, with a stable id (`F-001`, `F-002`, ...).
- Types: `bug`, `docs`, `dx` (developer experience), `gap`, `proposal`.
- Severity: `low`, `medium`, `high`.
- Status: `observation` (seen once, not reproduced), `confirmed` (reproduced on a clean install),
  `filed` (an upstream issue exists), `closed` (fixed or declined upstream).
- Keep the reproduction minimal: a command or a file and line, then expected and actual.
- Do not speculate about the fix in the `Observed` line; put it in `Proposal`.

## Open entries

| Id | Type | Severity | Area | Status | Upstream |
|---|---|---|---|---|---|
| F-001 | docs | low | README, AGENTS.md | confirmed | — |
| F-002 | dx | low | public exports | confirmed | — |
| F-003 | gap | medium | traces | confirmed | — |
| F-004 | dx | low | dataset examples | confirmed | — |
| F-005 | dx | low | `frontedge check` | observation | — |
| F-006 | proposal | low | per-example evidence | observation | — |

## Entries

### F-001: Documentation says four screens, lists five

**Area:** README, AGENTS.md
**Observed:** `README.md` describes "four screens" while "What you get" lists five: Runs, Run
detail, Compare, Eval diff, Dataset. `AGENTS.md` repeats "four screens".
**Expected:** The count matches the list, or the text does not count at all.
**Reproduction:** Open the README section "What you get" and compare it with the status paragraph.
**Workaround:** None needed for consumers.
**Proposal:** Replace the number with the list of screen names, so the sentence cannot drift again.

### F-002: Query and page models are not exported from the package root

**Area:** public exports
**Observed:** `frontedge.__all__` exports `ExamplesPage` and `DatasetExamplesPage`, but not
`RunsPage`, `RunsQuery` or `ExamplesQuery`. The integration guide imports them from
`frontedge.models`, so both paths exist and only some are top-level.
**Expected:** Every public model needed to implement a provider is available from `frontedge`, or
the guide uses one import path consistently.
**Reproduction:**

```powershell
python -c "import frontedge; print(hasattr(frontedge, 'RunsQuery'))"
# False
python -c "from frontedge.models import RunsQuery; print(RunsQuery.__name__)"
# RunsQuery
```

**Workaround:** Import from `frontedge.models` (what this repository does).
**Proposal:** Re-export the query and page models from `frontedge/__init__.py` and mention them in
the guide, matching the existing `ExamplesPage` / `DatasetExamplesPage` exports.

### F-003: Traces are advertised but there is no trace screen

**Area:** product surface
**Observed:** The repository description says "runs, datasets, evals and traces", and the design
system has a `trace` pattern, but "What you get" lists no traces screen and there is no
`/frontedge/api/traces` route.
**Expected:** Either traces are part of the mounted screens, or the description and vocabulary do
not promise them.
**Reproduction:** Compare the repository description and `AGENTS.md` ("it knows about run, dataset,
eval, trace") with the route table in `docs/integration.md`.
**Workaround:** Store trace-like evidence in per-example rows or in a dataset.
**Proposal:** Decide the status of traces: a screen with its own provider protocol, or a roadmap
note that removes the term from the description until the screen exists.

### F-004: Dataset rows reuse the comparison row model

**Area:** dataset examples
**Observed:** `DatasetExamplesPage.items` is `list[ExampleResult]`, where `ExampleResult` carries
`changed: bool`, `reference_output` / `comparison_output` and `reference_score` /
`comparison_score`. A dataset browser has no reference and no comparison, so it must invent
`changed=False` and leave the comparison fields empty.
**Expected:** A dataset row says what it is and does not require comparison-only fields.
**Reproduction:** See `ExampleResult` and `DatasetExamplesPage` in `python/frontedge/src/frontedge/models.py`.
**Workaround:** `ExampleResult(..., changed=False)` (see `providers/datasets.py` in this repository).
**Proposal:** Add a dedicated dataset row model (input, output, expected, optional score, tags), or
split `ExampleResult` into a base row plus comparison fields.

### F-005: `frontedge check --provider` assumes an importable provider object

**Area:** CLI
**Observed:** The documented check imports `module:object`. Applications that build providers
inside an app factory (`create_app`) or from settings have no obvious module-level object to point
at, so the check cannot be used without adding a test-only module.
**Expected:** A documented way to check a provider constructed at runtime.
**Reproduction:** Follow "Check the provider before the server starts" in `docs/integration.md` and
try it on a provider created inside a function.
**Workaround:** Start the server and call `/frontedge/api/runs`, or expose a scratch
`module:provider`.
**Proposal:** Accept a zero-argument factory (`module:factory` with `--factory`) or document the
`module:provider` pattern as the supported workaround.

### F-006: One score pair per example limits multi-metric RAG evidence

**Area:** per-example evidence
**Observed:** `EvalComparison` compares one metric per eval (`reference_score` /
`comparison_score`). A RAG example naturally has retrieval evidence (recall, MRR) and generation
evidence (F1, faithfulness) at the same time; the screen can only show one pair per eval id.
**Expected:** Either per-example evidence supports several named scores, or the one-metric-per-eval
model is documented as the intended composition rule.
**Reproduction:** Compare `EvalSummary.metric` with `ExampleResult.reference_score` in the models.
**Workaround:** Declare one eval id per metric; the eval picker then switches between them (what
this repository plans to do).
**Proposal:** Allow an optional `scores: dict[str, float]` on the example row, or state the
composition rule in the integration guide so providers do not look for a capability that is absent.

## Validated, no action

- The PyPI wheel ships the built interface: `uvicorn app:app` serves `/frontedge` with no Node at
  runtime.
- `frontedge check frontedge.toml` catches unknown keys and prints the effective configuration.
- The provider contract is clear enough to implement all three sources without reading the router
  source: runs, evals and datasets mounted from one `frontedge.mount(...)` call.
- `run_columns` accepts custom metric ids, and the configured metrics drive the cards and charts.

## Filing an issue

The helper creates one issue in `socrabs151/frontedge-ml` from an entry, using `GITHUB_TOKEN` /
`GH_TOKEN` or the stored git credential:

```powershell
python scripts/file_frontedge_issue.py --entry F-002 --dry-run
python scripts/file_frontedge_issue.py --entry F-002
```

Put the returned URL into the `Upstream` column and change `Status` to `filed`.
