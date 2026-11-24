from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schemas import CorpusDoc, DatasetRecord, DatasheetEntry, Question

DOWNLOAD_TIMEOUT_SECONDS = 60
DOWNLOAD_ATTEMPTS = 3
CHUNK_SIZE = 1 << 20
MANIFEST_NAME = "manifest.json"

Opener = Callable[..., Any]


class DatasetError(RuntimeError):
    """Raised when a dataset cannot be fetched or converted."""


class ChecksumError(DatasetError):
    """Raised when a downloaded artifact does not match the pinned hash."""


@dataclass(frozen=True)
class Artifact:
    """One pinned source file of a dataset."""

    path: str
    url: str
    sha256: str
    size: int


@dataclass(frozen=True)
class DatasetSpec:
    """Everything needed to reproduce a public dataset locally."""

    id: str
    title: str
    version: str
    source: str
    license: str
    artifacts: tuple[Artifact, ...]
    datasheet: tuple[tuple[str, str], ...]


SCIFACT_COMMIT = "cf10ab6856b15b0e670ef8ae5dae4e266c12d035"
SCIFACT_BASE = f"https://huggingface.co/datasets/mteb/scifact/resolve/{SCIFACT_COMMIT}"

SPECS: dict[str, DatasetSpec] = {
    "scifact": DatasetSpec(
        id="scifact",
        title="SciFact (BEIR): scientific claim retrieval",
        version=f"mteb@{SCIFACT_COMMIT[:7]}",
        source="https://huggingface.co/datasets/mteb/scifact",
        license="CC BY-SA 4.0 (SciFact by Wadden et al., 2020; BEIR distribution)",
        artifacts=(
            Artifact(
                path="_source/corpus.jsonl",
                url=f"{SCIFACT_BASE}/corpus.jsonl",
                sha256="f0d32db0d156b526d75921ed7a76f2cb912902631c87248c5c97c617bad0b60c",
                size=8023638,
            ),
            Artifact(
                path="_source/queries.jsonl",
                url=f"{SCIFACT_BASE}/queries.jsonl",
                sha256="9db7df096f7414435d52bafcbacf814c30cea50eb565c1e8fa6d11440759bba8",
                size=129085,
            ),
            Artifact(
                path="_source/qrels/test.tsv",
                url=f"{SCIFACT_BASE}/qrels/test.tsv",
                sha256="0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c",
                size=5389,
            ),
        ),
        datasheet=(
            (
                "Motivation",
                "Public retrieval benchmark for scientific claim verification, "
                "comparable with published BEIR results.",
            ),
            (
                "Composition",
                "5183 abstracts, 1109 claims in total, 300 claims in the test split "
                "with relevance judgements.",
            ),
            (
                "Collection",
                "SciFact (Wadden et al., 2020) through the mteb mirror, pinned by "
                "commit and checked by SHA-256.",
            ),
            (
                "Intended use",
                "Retrieval quality: Recall@k and MRR. The dataset ships no gold "
                "answers, so answer metrics are omitted.",
            ),
            (
                "Limitations",
                "English biomedical abstracts only; claims are short and lexical "
                "overlap with the target abstract is often high.",
            ),
        ),
    )
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, artifact: Artifact) -> None:
    size = path.stat().st_size
    if size != artifact.size:
        raise ChecksumError(
            f"{path.name}: expected {artifact.size} bytes, got {size}. "
            "Delete the file or re-run with --force."
        )
    actual = _sha256(path)
    if actual != artifact.sha256:
        raise ChecksumError(
            f"{path.name}: expected sha256 {artifact.sha256}, got {actual}. "
            "The upstream file changed; update the spec deliberately."
        )


def _download(
    url: str,
    destination: Path,
    *,
    opener: Opener = urllib.request.urlopen,
    attempts: int = DOWNLOAD_ATTEMPTS,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with opener(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                with partial.open("wb") as handle:
                    shutil.copyfileobj(response, handle, CHUNK_SIZE)
            os.replace(partial, destination)
            return
        except OSError as error:
            last_error = error
            partial.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(attempt)
    raise DatasetError(f"cannot download {url}: {last_error}")


def ensure_artifacts(
    spec: DatasetSpec,
    destination: Path,
    *,
    force: bool = False,
    opener: Opener = urllib.request.urlopen,
) -> list[Path]:
    """Download missing artifacts, verify every file against the pinned hash."""

    paths: list[Path] = []
    for artifact in spec.artifacts:
        target = destination / artifact.path
        if force or not target.is_file():
            _download(artifact.url, target, opener=opener)
        _verify(target, artifact)
        paths.append(target)
    return paths


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _sort_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):012d}") if value.isdigit() else (1, value)


def _read_qrels(path: Path) -> dict[str, list[str]]:
    qrels: dict[str, list[str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if int(row["score"]) <= 0:
                continue
            query_id = str(row["query-id"])
            corpus_id = str(row["corpus-id"])
            qrels.setdefault(query_id, [])
            if corpus_id not in qrels[query_id]:
                qrels[query_id].append(corpus_id)
    return qrels


def convert_scifact(
    source_dir: Path, destination: Path, spec: DatasetSpec
) -> tuple[DatasetRecord, int, int]:
    """Convert the raw SciFact files into the store layout used by the bench."""

    corpus = [
        CorpusDoc(id=str(row["_id"]), title=row.get("title", ""), text=row["text"])
        for row in _read_jsonl(source_dir / "_source" / "corpus.jsonl")
    ]
    corpus.sort(key=lambda doc: _sort_key(doc.id))

    queries = {
        str(row["_id"]): str(row["text"])
        for row in _read_jsonl(source_dir / "_source" / "queries.jsonl")
    }
    qrels = _read_qrels(source_dir / "_source" / "qrels" / "test.tsv")

    questions = [
        Question(
            id=query_id,
            question=text,
            expected_answer="",
            dataset="scifact",
            gold_doc_ids=sorted(qrels[query_id], key=_sort_key),
            tags=["beir", "scifact", "retrieval"],
        )
        for query_id, text in queries.items()
        if qrels.get(query_id)
    ]
    questions.sort(key=lambda item: _sort_key(item.id))

    destination.mkdir(parents=True, exist_ok=True)
    _write_jsonl(destination / "corpus.jsonl", [doc.model_dump() for doc in corpus])
    _write_jsonl(destination / "questions.jsonl", [q.model_dump() for q in questions])

    record = DatasetRecord(
        id=spec.id,
        name=spec.title,
        version=spec.version,
        created_at=datetime.now(UTC),
        tags=["beir", "scifact", "retrieval", "english"],
        datasheet=[DatasheetEntry(label=label, value=value) for label, value in spec.datasheet],
        provenance=[
            DatasheetEntry(label="Source", value=spec.source, mono=True),
            DatasheetEntry(label="Version", value=spec.version, mono=True),
            DatasheetEntry(label="License", value=spec.license),
            DatasheetEntry(
                label="Files",
                value="corpus.jsonl, queries.jsonl, qrels/test.tsv",
                mono=True,
            ),
        ],
    )
    (destination / "datasets.json").write_text(
        json.dumps([record.model_dump(mode="json")], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return record, len(corpus), len(questions)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(
    spec: DatasetSpec,
    destination: Path,
    *,
    documents: int,
    questions: int,
) -> Path:
    files = []
    for artifact in spec.artifacts:
        target = destination / artifact.path
        files.append(
            {
                "path": artifact.path,
                "url": artifact.url,
                "sha256": _sha256(target),
                "size": target.stat().st_size,
            }
        )
    manifest = {
        "dataset": spec.id,
        "version": spec.version,
        "source": spec.source,
        "license": spec.license,
        "fetched_at": datetime.now(UTC).isoformat(),
        "documents": documents,
        "questions": questions,
        "files": files,
    }
    path = destination / MANIFEST_NAME
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


Converter = Callable[[Path, Path, DatasetSpec], tuple[DatasetRecord, int, int]]

CONVERTERS: dict[str, Converter] = {
    "scifact": convert_scifact,
}


def read_manifest(destination: Path) -> dict[str, Any]:
    path = destination / MANIFEST_NAME
    if not path.is_file():
        raise DatasetError(f"no manifest in {destination}; run fetch-data first")
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_dataset(
    name: str,
    destination: Path | str,
    *,
    force: bool = False,
    opener: Opener = urllib.request.urlopen,
) -> tuple[DatasetRecord, int, int]:
    """Fetch, verify and convert a public dataset into ``destination``."""

    spec = SPECS.get(name)
    converter = CONVERTERS.get(name)
    if spec is None or converter is None:
        known = ", ".join(sorted(SPECS))
        raise DatasetError(f"unknown dataset {name!r}; available: {known}")

    target = Path(destination)
    ensure_artifacts(spec, target, force=force, opener=opener)
    record, documents, questions = converter(target, target, spec)
    write_manifest(spec, target, documents=documents, questions=questions)
    return record, documents, questions
