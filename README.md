# rag-eval-bench

A reproducible bench for retrieval-augmented generation: retrieval ablations, answer quality,
and a ready-made interface for runs, evals and datasets.

**Status: skeleton.** It runs on bundled demo data and produces real experiment runs with BM25
and a deterministic extractive generator. Dense retrieval, rerankers, local LLM generation and
judge calibration are wired as extension points, not finished features.

## Why

Most RAG projects rebuild the same screens: a table of runs, metric curves, a comparison of two
evaluations, a dataset browser. That plumbing is not the research. This repository keeps the
experiment as the product and uses [frontedge](https://github.com/socrabs151/frontedge-ml) as the
interface layer: a FastAPI router with runs, run detail, compare, eval diff and datasets, fed by
three small providers.

The UI is a dependency, not the subject. The subject is: which retrieval configuration is better,
how much does it cost, and can the evaluation be trusted.

## What works

- **BM25** over overlapping character chunks, implemented in pure Python.
- **Extractive baseline generator**: returns the context sentence closest to the question, so the
  pipeline is deterministic and runs offline.
- **Ollama generator**: local models through the chat API, used when you pass
  `--generator ollama`.
- **Metrics**: Recall@k, MRR, token F1, exact match.
- **Experiment runner**: evaluates every question, aggregates metrics and writes one JSON run to
  `artifacts/runs/`.
- **frontedge providers**:
  - runs: server-side filtering, sorting and pagination, per-run examples;
  - evals: per-example comparison of two runs;
  - datasets: versions, datasheet and provenance.
- **Demo data**: 12 glossary notes, 8 questions, 3 finished runs with per-example evidence.

## Layout

```text
app.py                     uvicorn entry point
frontedge.toml             columns, metrics, baseline, page size
data/                      demo corpus, questions, runs, evals, datasets
scripts/build_demo_data.py regenerates data/ from one source of truth
src/rag_eval_bench/
  app.py                   FastAPI + frontedge.mount
  providers/               runs / evals / datasets adapters
  retrieval/               bm25, dense, hybrid (RRF)
  generation/              extractive baseline, Ollama client
  evaluation/              metrics and the experiment runner
  storage.py               JSON store for corpus and runs
tests/                     metrics, chunking, retrieval, providers, API
```

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
frontedge check frontedge.toml
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000/frontedge>. The API lives under `/frontedge/api`.

Run an experiment and restart the server to see it:

```powershell
rag-eval-bench run --retriever bm25 --k 5
```

The command writes `artifacts/runs/<run-id>.json`. `artifacts/` is gitignored; the demo runs in
`data/runs/` are committed.

## Data

`data/` is generated. Edit `scripts/build_demo_data.py` and run it instead of editing JSON by
hand:

```powershell
python scripts/build_demo_data.py
```

The corpus is English, written for this repository. It is a smoke-test fixture, not a benchmark:
eight questions are too few for comparative claims. Replace it with a real corpus before drawing
conclusions.

## How the evaluation works

For every question the runner retrieves `k` chunks, generates an answer and records:

- `recall_at_5`: whether a gold document reached the top 5;
- `mrr_at_10`: the reciprocal rank of the first gold document;
- `f1`: token overlap between the answer and the expected answer;
- `latency_ms`: retrieval plus generation time.

Aggregates are means over questions; the series are cumulative means by step, so the run detail
has a curve. Every example keeps the question, both answers, the retrieved document ids and its
score, which is what the eval diff screen compares row by row.

`faithfulness` is reserved for the judge stage: the demo data stores a value from a previous
annotation pass, the runner writes `0.0` until a judge is connected.

## Frontedge feedback

Findings about the dependency are logged in
[`docs/frontedge-feedback.md`](docs/frontedge-feedback.md): each entry has a minimal reproduction,
a workaround and a proposed change. To file one upstream:

```powershell
python scripts/file_frontedge_issue.py --entry F-002 --dry-run
python scripts/file_frontedge_issue.py --entry F-002
```

The helper targets `socrabs151/frontedge-ml` and uses `GITHUB_TOKEN` or the stored git credential.
After filing, put the issue URL into the `Upstream` column of the log.

## Roadmap

1. Dense retrieval (`sentence-transformers`) and a cross-encoder reranker in the hybrid path.
2. A generator matrix: 1.5B / 3B / 7B through Ollama, plus one hosted model for the Pareto point.
3. LLM-as-a-judge with a human-labelled subset and Cohen's kappa calibration.
4. Bootstrap confidence intervals and paired tests on per-example scores.
5. Corpus ingestion for a real document collection and dataset versioning.

## Notes

- `frontedge` is pinned to `0.5.x` (public beta). Keep the providers behind this repository's
  thin adapters, so a library change stays local.
- Python 3.11+; the demo needs no GPU and no network.
- For local models on Windows/AMD, Ollama is optional. The extractive baseline always works.
- A large model cache does not belong on a small system drive: point `HF_HOME` and
  `OLLAMA_MODELS` at a data disk.
