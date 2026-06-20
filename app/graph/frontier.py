from typing import List, Dict, Set
from dataclasses import dataclass


@dataclass
class Frontier:
    node_ids: List[int]
    chunk_ids: Set[int]
    entity_ids: Set[int]
    num_edges: int


def expand_frontier(
    seed_chunk_ids: Set[int],
    entity_to_chunks: Dict[int, List[int]],
    chunk_to_entities: Dict[int, List[int]],
    max_nodes: int = 50000,
    max_edges: int = 250000,
    max_hops: int = 2,
    max_entity_degree: int = 500,
) -> Frontier:
    visited_chunks: Set[int] = set(seed_chunk_ids)
    visited_entities: Set[int] = set()
    frontier_chunks: Set[int] = set(seed_chunk_ids)
    frontier_entities: Set[int] = set()
    total_edges = 0

    current_chunks = set(seed_chunk_ids)

    for hop in range(max_hops):
        next_entities: Set[int] = set()

        for chunk_id in current_chunks:
            if len(frontier_chunks) + len(frontier_entities) >= max_nodes:
                break
            if total_edges >= max_edges:
                break

            ents = chunk_to_entities.get(chunk_id, [])
            for e in ents:
                if e not in visited_entities:
                    degree = len(entity_to_chunks.get(e, []))
                    if degree <= max_entity_degree:
                        next_entities.add(e)
                        visited_entities.add(e)
                        total_edges += 1
                        if len(frontier_chunks) + len(frontier_entities) + len(next_entities) >= max_nodes:
                            break
                if total_edges >= max_edges:
                    break

        frontier_entities.update(next_entities)

        if len(frontier_chunks) + len(frontier_entities) >= max_nodes:
            break
        if total_edges >= max_edges:
            break

        next_chunks: Set[int] = set()
        for e in next_entities:
            if len(frontier_chunks) + len(frontier_entities) + len(next_chunks) >= max_nodes:
                break
            if total_edges >= max_edges:
                break

            for c in entity_to_chunks.get(e, []):
                if c not in visited_chunks:
                    next_chunks.add(c)
                    visited_chunks.add(c)
                    total_edges += 1
                    if len(frontier_chunks) + len(frontier_entities) + len(next_chunks) >= max_nodes:
                        break
                if total_edges >= max_edges:
                    break

        frontier_chunks.update(next_chunks)
        current_chunks = next_chunks

        if len(frontier_chunks) + len(frontier_entities) >= max_nodes:
            break
        if total_edges >= max_edges:
            break

    all_nodes = sorted(frontier_chunks | frontier_entities)
    return Frontier(
        node_ids=all_nodes,
        chunk_ids=frontier_chunks,
        entity_ids=frontier_entities,
        num_edges=total_edges,
    )
