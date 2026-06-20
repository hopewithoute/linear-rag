import pytest
from app.retrieval.packer import pack_contexts


def test_pack_contexts_basic():
    chunks = [
        {"chunk_id": 1, "document_id": 10, "score": 0.9, "text": "hello", "evidence_path": []},
        {"chunk_id": 2, "document_id": 10, "score": 0.8, "text": "world", "evidence_path": []},
        {"chunk_id": 3, "document_id": 11, "score": 0.7, "text": "foo", "evidence_path": []},
    ]
    packed = pack_contexts(chunks, top_k=2)
    assert len(packed) == 2
    assert packed[0]["score"] >= packed[1]["score"]


def test_pack_contexts_dedup():
    chunks = [
        {"chunk_id": 1, "document_id": 10, "score": 0.9, "text": "same", "evidence_path": []},
        {"chunk_id": 2, "document_id": 10, "score": 0.8, "text": "same", "evidence_path": []},
    ]
    packed = pack_contexts(chunks, top_k=10)
    assert len(packed) == 1


def test_pack_contexts_max_per_doc():
    chunks = [
        {"chunk_id": i, "document_id": 10, "score": 1.0 - i * 0.1, "text": f"text_{i}", "evidence_path": []}
        for i in range(10)
    ]
    packed = pack_contexts(chunks, top_k=10, max_per_doc=3)
    assert len(packed) == 3
