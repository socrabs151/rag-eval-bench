# rag-eval-bench

[![ci](https://github.com/socrabs151/rag-eval-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/socrabs151/rag-eval-bench/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

A reproducible bench for retrieval-augmented generation: retrieval ablations, answer quality,
and a ready-made interface for runs, evals and datasets.

**Status: working, pre-judge.** BM25, dense and hybrid retrieval plus cross-encoder reranking run
on the bundled demo corpus and on SciFact; the extractive baseline and Ollama generation are
wired; retrieval metrics, reranking and the run store are covered by tests and CI. LLM-as-a-judge
and statistical tests are the next milestones (see [Roadmap](#roadmap)).

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
- **Cross-encoder reranking** of the retrieved candidates (`ms-marco-mini` or `bge-base`), with
  retrieval and rerank latency recorded separately.
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

# dense and reranked runs need the optional extra
pip install -e ".[retrieval]"
rag-eval-bench run --data-dir datasets/scifact --retriever hybrid --k 5 \
    --reranker ms-marco-mini --candidates 20

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

### Ablation

`scripts/run_ablation.py` runs four configurations through the public runner and writes
[`reports/scifact-ablation.md`](reports/scifact-ablation.md) together with a JSON copy. The table
below is that report, regenerated as a whole; nothing is typed by hand.

| Retrieval | Recall@5 | MRR@10 | retrieval, ms | rerank, ms | total p50, ms | total p95, ms |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.650 | 0.545 | 143 | — | 138 | 194 |
| Dense (multilingual e5-small) | 0.623 | 0.534 | 18 | — | 18 | 22 |
| Hybrid (RRF) | 0.703 | 0.599 | 232 | — | 218 | 384 |
| Hybrid + cross-encoder | 0.747 | 0.625 | 245 | 257 | 485 | 680 |

Setup: 300 judged claims, `k=5`, 20 reranker candidates, 400-character chunks with 50 overlap, the
extractive generator. Latency is mean per query, the index build is excluded, and the totals are
p50/p95 over examples; the machine is CPU-only, so wall-clock will differ elsewhere.

What the numbers say: hybrid RRF beats both first-stage retrievers, and the cross-encoder adds
4.4 recall points at roughly twice the latency. The small multilingual dense model is slightly
worse than BM25 on English biomedical abstracts, which is the expected result for this corpus and
a reason to try a larger or domain-tuned encoder next.

Full reproduction takes about 30 minutes on CPU:

```powershell
python scripts/run_ablation.py --data-dir datasets/scifact --out reports/scifact-ablation.md
```

These are local reference numbers, not a leaderboard entry: BEIR reports nDCG@10 over full
documents, while here chunking and latency depend on the machine.

## How the evaluation works

For every question the runner retrieves `k` chunks, generates an answer and records:

- `recall_at_5`: whether a gold document reached the top 5;
- `mrr_at_10`: the reciprocal rank of the first gold document;
- `f1`: token overlap between the answer and the expected answer, when one exists;
- `retrieval_ms` and `rerank_ms`: mean stage latency, recorded when a reranker is used;
- `latency_ms`: retrieval, reranking and generation together.

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

1. LLM-as-a-judge with a human-labelled subset and Cohen's kappa calibration.
2. Bootstrap confidence intervals and paired tests on per-example scores.
3. Token-based chunking and a larger dense encoder, re-measured with the same ablation.
4. A generator matrix (local 1.5B–7B and one hosted model) with token cost per 1 000 queries.
5. Corpus ingestion for a real document collection and dataset versioning.

## Notes

- `frontedge` is pinned to `0.5.x` (public beta). Keep the providers behind this repository's
  thin adapters, so a library change stays local.
- Python 3.11+; the demo needs no GPU and no network.
- For local models on Windows/AMD, Ollama is optional. The extractive baseline always works.
- A large model cache does not belong on a small system drive: point `HF_HOME` and
  `OLLAMA_MODELS` at a data disk.
