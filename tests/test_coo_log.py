import pytest
from app.graph.coo_log import COOLog


def test_coo_log_append():
    log = COOLog()
    log.append(row=0, col=1, value=1.0)
    assert len(log) == 1
    assert log.rows == [0]
    assert log.cols == [1]
    assert log.values == [1.0]


def test_coo_log_bidirectional():
    log = COOLog()
    log.append_bidirectional(node_a=10, node_b=100, value=0.8)
    assert len(log) == 2
    assert log.rows == [10, 100]
    assert log.cols == [100, 10]


def test_coo_log_to_arrays():
    log = COOLog()
    log.append(row=0, col=5, value=0.5)
    log.append(row=5, col=0, value=0.5)
    rows, cols, vals = log.to_arrays()
    assert rows.tolist() == [0, 5]
    assert cols.tolist() == [5, 0]
    assert vals.tolist() == [0.5, 0.5]
