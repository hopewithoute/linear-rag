import pytest
import numpy as np
from app.graph.compiler import compute_entity_idf, apply_hub_penalty, normalize_columns, compile_snapshot
from app.graph.coo_log import COOLog


def test_compute_entity_idf():
    entity_doc_freq = {1: 10, 2: 100, 3: 1}
    total_chunks = 1000
    idf = compute_entity_idf(entity_doc_freq, total_chunks)
    assert idf[1] > idf[2]
    assert idf[3] > idf[1]


def test_apply_hub_penalty():
    rows = np.array([100, 100, 101], dtype=np.int32)
    cols = np.array([0, 1, 2], dtype=np.int32)
    values = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    entity_idf = {0: 2.0, 1: 0.5}
    entity_degree = {0: 10, 1: 1000}
    chunk_count = 100

    adjusted = apply_hub_penalty(rows, cols, values, entity_idf, entity_degree, chunk_count)
    assert adjusted[0] < 1.0
    assert adjusted[2] < adjusted[0]


def test_compile_snapshot_empty():
    coo = COOLog()
    matrix = compile_snapshot(coo, chunk_count=10, entity_count=5, entity_doc_freq={}, entity_degree={})
    assert matrix.shape == (15, 15)


def test_compile_snapshot_basic():
    coo = COOLog()
    coo.append_bidirectional(0, 3, 1.0)
    coo.append_bidirectional(1, 3, 1.0)
    coo.append_bidirectional(2, 4, 1.0)

    matrix = compile_snapshot(
        coo, chunk_count=3, entity_count=2,
        entity_doc_freq={0: 2, 1: 1},
        entity_degree={0: 3, 1: 1},
    )
    assert matrix.shape == (5, 5)
    assert matrix.nnz > 0
