from rag_eval_bench.chunking import Chunk
from rag_eval_bench.retrieval.base import RetrievedChunk
from rag_eval_bench.retrieval.bm25 import BM25Retriever
from rag_eval_bench.retrieval.hybrid import reciprocal_rank_fusion
from rag_eval_bench.schemas import CorpusDoc


def _chunk(doc_id: str, text: str) -> Chunk:
    return Chunk(id=f"{doc_id}#0", doc_id=doc_id, title=doc_id, text=text)


def test_bm25_ranks_the_gold_document_first() -> None:
    chunks = [
        _chunk("rag", "Retrieval-augmented generation combines a retriever and a generator."),
        _chunk("bm25", "BM25 uses the parameters k1 and b to rank documents."),
        _chunk("dense", "Dense retrieval maps queries and passages into a vector space."),
    ]
    hits = BM25Retriever(chunks).search("Which parameters does BM25 use?", 3)
    assert hits
    assert hits[0].chunk.doc_id == "bm25"


def test_bm25_ignores_queries_without_known_terms() -> None:
    chunks = [_chunk("rag", "Retrieval-augmented generation combines a retriever.")]
    assert BM25Retriever(chunks).search("zzz", 5) == []


def test_rrf_prefers_documents_ranked_by_both_retrievers() -> None:
    first = RetrievedChunk(chunk=_chunk("a", "a"), score=3.0)
    second = RetrievedChunk(chunk=_chunk("b", "b"), score=2.0)
    rankings = [[first, second], [first]]
    fused = reciprocal_rank_fusion(rankings)
    assert [hit.chunk.doc_id for hit in fused] == ["a", "b"]


def test_chunk_documents_is_importable_with_the_corpus_model() -> None:
    assert CorpusDoc(id="x", title="X", text="text").id == "x"
