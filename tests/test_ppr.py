import pytest
import numpy as np
from scipy import sparse
from app.graph.ppr import run_ppr


def _make_simple_matrix():
    data = np.array([0.5, 0.5, 0.5, 0.5, 1.0], dtype=np.float32)
    rows = np.array([0, 1, 1, 0, 2], dtype=np.int32)
    cols = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    return sparse.coo_matrix((data, (rows, cols)), shape=(3, 3)).tocsr()


def test_ppr_seed_chunks_highest_score():
    matrix = _make_simple_matrix()
    scores, paths = run_ppr(matrix, seed_scores={0: 1.0}, chunk_nodes={0, 1, 2}, alpha=0.15, max_iter=3)
    assert scores.get(0, 0) > scores.get(1, 0)


def test_ppr_inactive_chunks_masked():
    matrix = _make_simple_matrix()
    active_mask = np.array([True, False, True], dtype=bool)
    scores, paths = run_ppr(matrix, seed_scores={0: 1.0}, chunk_nodes={0, 1, 2}, active_mask=active_mask, alpha=0.15, max_iter=3)
    assert scores.get(1, 0) == 0.0


def test_ppr_empty_seed():
    matrix = _make_simple_matrix()
    scores, paths = run_ppr(matrix, seed_scores={}, chunk_nodes={0, 1, 2}, alpha=0.15, max_iter=3)
    assert len(scores) == 0


def test_ppr_returns_paths():
    matrix = _make_simple_matrix()
    scores, paths = run_ppr(matrix, seed_scores={0: 1.0}, chunk_nodes={0, 1, 2}, alpha=0.15, max_iter=3)
    assert isinstance(paths, dict)
    assert 0 in paths  # seed node should have path
