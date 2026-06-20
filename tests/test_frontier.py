import pytest
from app.graph.frontier import expand_frontier


def test_expand_frontier_simple():
    entity_to_chunks = {100: [0, 1], 101: [1, 2]}
    chunk_to_entities = {0: [100], 1: [100, 101], 2: [101]}

    frontier = expand_frontier(
        seed_chunk_ids={0},
        entity_to_chunks=entity_to_chunks,
        chunk_to_entities=chunk_to_entities,
        max_nodes=100,
        max_edges=100,
        max_hops=2,
    )
    assert 0 in frontier.chunk_ids
    assert 100 in frontier.entity_ids
    assert 1 in frontier.chunk_ids


def test_expand_frontier_respects_limits():
    entity_to_chunks = {i: list(range(100)) for i in range(200)}
    chunk_to_entities = {i: list(range(200)) for i in range(100)}

    frontier = expand_frontier(
        seed_chunk_ids={0},
        entity_to_chunks=entity_to_chunks,
        chunk_to_entities=chunk_to_entities,
        max_nodes=50,
        max_edges=100,
        max_hops=2,
    )
    assert len(frontier.chunk_ids) + len(frontier.entity_ids) <= 50


def test_expand_frontier_respects_entity_degree():
    entity_to_chunks = {100: list(range(600))}
    chunk_to_entities = {0: [100]}

    frontier = expand_frontier(
        seed_chunk_ids={0},
        entity_to_chunks=entity_to_chunks,
        chunk_to_entities=chunk_to_entities,
        max_nodes=1000,
        max_edges=1000,
        max_hops=1,
        max_entity_degree=500,
    )
    assert 100 not in frontier.entity_ids
