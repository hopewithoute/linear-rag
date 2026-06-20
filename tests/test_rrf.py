import pytest
from app.retrieval.rrf import reciprocal_rank_fusion


def test_rrf_basic():
    dense = [{"chunk_id": 1}, {"chunk_id": 2}, {"chunk_id": 3}]
    bm25 = [{"chunk_id": 2}, {"chunk_id": 3}, {"chunk_id": 4}]

    scores = reciprocal_rank_fusion(dense, bm25, w_dense=0.6, w_bm25=0.4, k=60)
    assert scores[2] > scores[1]
    assert scores[3] > scores[1]
    assert 4 in scores


def test_rrf_single_source():
    dense = [{"chunk_id": 1}, {"chunk_id": 2}]
    bm25 = []

    scores = reciprocal_rank_fusion(dense, bm25)
    assert scores[1] > scores[2]


def test_rrf_empty():
    scores = reciprocal_rank_fusion([], [])
    assert len(scores) == 0
