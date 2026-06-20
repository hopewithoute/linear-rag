"""BFS retrieval mode inspired by LinearRAG.
Uses BFS with semantic bridging for multi-hop reasoning.
"""
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class BFSResult:
    chunk_ids: List[int]
    scores: Dict[int, float]
    evidence_paths: Dict[int, List[str]]
    iterations: int


def _run_bfs_expansion(
    seed_scores: Dict[int, float],
    chunk_to_entities: Dict[int, List[int]],
    entity_to_chunks: Dict[int, List[int]],
    score_fn: Callable[[List[int]], Dict[int, float]],
    max_iterations: int = 3,
    iteration_threshold: float = 0.4,
    max_chunks: int = 1000,
    max_entity_degree: int = 500,
) -> BFSResult:
    visited_chunks: Set[int] = set(seed_scores.keys())
    visited_entities: Set[int] = set()
    
    current_chunks = set(seed_scores.keys())
    all_scores: Dict[int, float] = dict(seed_scores)
    evidence_paths: Dict[int, List[str]] = {
        cid: [f"query -> chunk_{cid}"] for cid in seed_scores
    }
    
    iterations = 0
    
    for iteration in range(max_iterations):
        iterations = iteration + 1
        
        next_entities: Set[int] = set()
        for chunk_id in current_chunks:
            entities = chunk_to_entities.get(chunk_id, [])
            for e in entities:
                if e not in visited_entities:
                    degree = len(entity_to_chunks.get(e, []))
                    if degree <= max_entity_degree:
                        next_entities.add(e)
                        visited_entities.add(e)
        
        if not next_entities:
            break
        
        next_chunks: Set[int] = set()
        for e in next_entities:
            for c in entity_to_chunks.get(e, []):
                if c not in visited_chunks:
                    next_chunks.add(c)
                    visited_chunks.add(c)
                    evidence_paths[c] = [f"query -> chunk_{c} (via entity_{e})"]
        
        if not next_chunks:
            break
        
        new_scores = score_fn(list(next_chunks))
        
        # Check iteration threshold if we have variable scores
        if new_scores:
            max_new_score = max(new_scores.values())
            # We bypass the threshold check if max_new_score is exactly 0.1 (uniform fallback logic in bfs_retrieve)
            if max_new_score != 0.1 and max_new_score < iteration_threshold:
                break
        
        all_scores.update(new_scores)
        current_chunks = next_chunks
        
        if len(all_scores) >= max_chunks:
            break
    
    ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    chunk_ids = [cid for cid, _ in ranked[:max_chunks]]
    
    return BFSResult(
        chunk_ids=chunk_ids,
        scores=all_scores,
        evidence_paths=evidence_paths,
        iterations=iterations,
    )


def bfs_retrieve(
    seed_scores: Dict[int, float],
    chunk_to_entities: Dict[int, List[int]],
    entity_to_chunks: Dict[int, List[int]],
    query_embedding: Optional[np.ndarray] = None,
    chunk_embeddings: Optional[Dict[int, np.ndarray]] = None,
    max_iterations: int = 3,
    iteration_threshold: float = 0.4,
    max_chunks: int = 1000,
    max_entity_degree: int = 500,
) -> BFSResult:
    """BFS retrieval with semantic bridging."""
    
    def score_fn(next_chunks: List[int]) -> Dict[int, float]:
        if query_embedding is not None and chunk_embeddings:
            return _compute_similarity_scores(next_chunks, query_embedding, chunk_embeddings)
        return {c: 0.1 for c in next_chunks}

    return _run_bfs_expansion(
        seed_scores=seed_scores,
        chunk_to_entities=chunk_to_entities,
        entity_to_chunks=entity_to_chunks,
        score_fn=score_fn,
        max_iterations=max_iterations,
        iteration_threshold=iteration_threshold,
        max_chunks=max_chunks,
        max_entity_degree=max_entity_degree,
    )


def _compute_similarity_scores(
    chunk_ids: List[int],
    query_embedding: np.ndarray,
    chunk_embeddings: Dict[int, np.ndarray],
) -> Dict[int, float]:
    """Compute cosine similarity between query and chunks."""
    scores = {}
    query_norm = np.linalg.norm(query_embedding)
    
    for cid in chunk_ids:
        emb = chunk_embeddings.get(cid)
        if emb is None:
            continue
        
        dot = np.dot(query_embedding, emb)
        norm = query_norm * np.linalg.norm(emb)
        scores[cid] = float(dot / norm) if norm > 0 else 0.0
    
    return scores


def bfs_with_rerank(
    seed_scores: Dict[int, float],
    chunk_to_entities: Dict[int, List[int]],
    entity_to_chunks: Dict[int, List[int]],
    query: str,
    chunk_texts: Dict[int, str],
    max_iterations: int = 3,
    iteration_threshold: float = 0.4,
    max_chunks: int = 1000,
    max_entity_degree: int = 500,
) -> BFSResult:
    """BFS retrieval with text-based reranking."""
    query_words = set(query.lower().split())

    def score_fn(next_chunks: List[int]) -> Dict[int, float]:
        return _compute_text_scores(next_chunks, query_words, chunk_texts)

    return _run_bfs_expansion(
        seed_scores=seed_scores,
        chunk_to_entities=chunk_to_entities,
        entity_to_chunks=entity_to_chunks,
        score_fn=score_fn,
        max_iterations=max_iterations,
        iteration_threshold=iteration_threshold,
        max_chunks=max_chunks,
        max_entity_degree=max_entity_degree,
    )


def _compute_text_scores(
    chunk_ids: List[int],
    query_words: Set[str],
    chunk_texts: Dict[int, str],
) -> Dict[int, float]:
    """Compute text-based similarity scores."""
    scores = {}
    
    for cid in chunk_ids:
        text = chunk_texts.get(cid, "").lower()
        if not text:
            scores[cid] = 0.0
            continue
        
        text_words = set(text.split())
        overlap = len(query_words & text_words)
        scores[cid] = overlap / len(query_words) if len(query_words) > 0 else 0.0
    
    return scores
