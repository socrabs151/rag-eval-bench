import pytest

from rag_eval_bench.chunking import chunk_documents
from rag_eval_bench.schemas import CorpusDoc


def test_chunking_splits_with_overlap() -> None:
    text = "abcdefghij" * 200
    chunks = chunk_documents([CorpusDoc(id="doc", title="Doc", text=text)], size=400, overlap=50)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 400 for chunk in chunks)
    assert chunks[0].id == "doc#0"
    assert chunks[1].id == "doc#1"
    assert chunks[0].text[-50:] in chunks[1].text


def test_chunking_keeps_a_short_document_whole() -> None:
    chunks = chunk_documents([CorpusDoc(id="doc", title="Doc", text="short text")])
    assert len(chunks) == 1
    assert chunks[0].text == "short text"


def test_chunking_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_documents([], size=100, overlap=100)
