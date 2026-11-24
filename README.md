# rag-eval-bench

[![ci](https://github.com/socrabs151/rag-eval-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/socrabs151/rag-eval-bench/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

A reproducible bench for retrieval-augmented generation: retrieval ablations, answer quality,
and a ready-made interface for runs, evals and datasets.

**Status: working, pre-judge.** BM25, dense and hybrid retrieval run on the bundled demo corpus
and on SciFact; the extractive baseline and Ollama generation are wired; retrieval metrics and
the run store are covered by tests and CI. LLM-as-a-judge, rerankers and statistical tests are
the next milestones (see [Roadmap](#roadmap)).

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
- **Dense retrieval** through `sentence-transformers` and **hybrid RRF** over both rankings.
- **Extractive baseline generator**: returns the context sentence closest to the question, so the
  pipeline is deterministic and runs offline.
- **Ollama generator**: local models through the chat API, used when you pass
  `--generator ollama`.
- **Metrics**: Recall@k, MRR, token F1, exact match; answer metrics are reported only when the
  dataset has reference answers.
- **Experiment runner**: evaluates every question, aggregates metrics and writes one JSON run to
  `artifacts/runs/`.
- **Dataset fetching**: `fetch-data` downloads a public benchmark, verifies every file against a
  pinned SHA-256 and converts it to the store layout.
- **frontedge providers**: runs with server-side filtering, sorting and pagination; evals with
  per-example comparison; datasets with versions, datasheet and provenance.

## Layout

```text
app.py                     uvicorn entry point
frontedge.toml             columns, metrics, baseline, page size
data/                      demo corpus, questions, runs, evals, datasets
datasets/                  fetched public datasets (gitignored)
scripts/build_demo_data.py regenerates data/ from one source of truth
src/rag_eval_bench/
  app.py                   FastAPI + frontedge.mount
  cli.py                   run, serve, fetch-data
  datasets.py              pinned specs, verified download, conversion
  providers/               runs / evals / datasets adapters
  retrieval/               bm25, dense, hybrid (RRF)
  generation/              extractive baseline, Ollama client
  evaluation/              metrics and the experiment runner
  storage.py               JSON store for corpus and runs
tests/                     metrics, chunking, retrieval, providers, API, datasets
```

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# demo corpus, no network
rag-eval-bench run --retriever bm25 --k 5

# public benchmark
rag-eval-bench fetch-data --dataset scifact
rag-eval-bench run --data-dir datasets/scifact --retriever bm25 --k 5

# interface
rag-eval-bench serve
```

Open <http://127.0.0.1:8000/frontedge>. The API lives under `/frontedge/api`.
`artifacts/` is gitignored; the demo runs in `data/runs/` are committed.

Docker is available too: `docker build -t rag-eval-bench .` then
`docker run --rm -p 8000:8000 rag-eval-bench`.

## Datasets

| id | source | size | gold answers | use |
|----|--------|------|--------------|-----|
| `scifact` | [SciFact (BEIR)](https://huggingface.co/datasets/mteb/scifact) | 5 183 abstracts, 300 judged claims | no | retrieval quality |
| `ml-glossary` (bundled) | this repository | 12 notes, 8 questions | yes | smoke tests, UI work |

`fetch-data` pins the upstream commit and the SHA-256 of every file, verifies them before
conversion and writes `manifest.json` next to the data. A changed upstream file fails the fetch
instead of silently changing the numbers. SciFact carries relevance judgements but no reference
answers, so F1 and faithfulness are omitted for it; answer quality is the judge milestone.

Licence: SciFact is distributed under CC BY-SA 4.0 (Wadden et al., 2020; BEIR distribution). The
data is downloaded at run time and is not committed here.

### Reference run

```text
command: rag-eval-bench run --data-dir datasets/scifact --retriever bm25 --k 5
chunks:  400 characters, 50 overlap
result:  recall@5 = 0.650, MRR@10 = 0.545, ~150 ms per query
```

These numbers are a local reference, not a leaderboard entry: BEIR reports nDCG@10 over full
documents, and the chunking and latency here depend on the machine.

## How the evaluation works

For every question the runner retrieves `k` chunks, generates an answer and records:

- `recall_at_5`: whether a gold document reached the top 5;
- `mrr_at_10`: the reciprocal rank of the first gold document;
- `f1`: token overlap between the answer and the expected answer, when one exists;
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

1. Cross-encoder reranker in the hybrid path, with a latency and quality comparison.
2. LLM-as-a-judge with a human-labelled subset and Cohen's kappa calibration.
3. Bootstrap confidence intervals and paired tests on per-example scores.
4. A generator matrix (local 1.5B–7B and one hosted model) with token cost per 1 000 queries.
5. Corpus ingestion for a real document collection and dataset versioning.

## Notes

- `frontedge` is pinned to `0.5.x` (public beta). Keep the providers behind this repository's
  thin adapters, so a library change stays local.
- Python 3.11+; the demo needs no GPU and no network.
- For local models on Windows/AMD, Ollama is optional. The extractive baseline always works.
- A large model cache does not belong on a small system drive: point `HF_HOME` and
  `OLLAMA_MODELS` at a data disk.
