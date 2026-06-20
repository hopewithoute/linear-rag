from typing import Dict, Set, Optional
from dataclasses import dataclass


@dataclass
class AdaptiveParams:
    alpha: float
    max_iter: int
    max_nodes: int
    max_edges: int
    max_hops: int


def compute_adaptive_params(
    query: str,
    seed_scores: Dict[int, float],
    corpus_size: int = 1000,
    base_alpha: float = 0.15,
    base_max_iter: int = 3,
    base_max_nodes: int = 50000,
    base_max_edges: int = 250000,
    base_max_hops: int = 2,
) -> AdaptiveParams:
    """Compute adaptive PPR parameters based on query and corpus characteristics.

    Strategy:
    - Simple queries (few words) → higher alpha (less exploration)
    - Complex queries (many words) → lower alpha (more exploration)
    - Small corpus → smaller frontier
    - Large corpus → larger frontier
    - More seed chunks → more exploration needed
    """
    query_words = query.split()
    num_words = len(query_words)
    num_seeds = len(seed_scores)

    # Adaptive alpha
    # Simple query → higher alpha (stay close to seeds)
    # Complex query → lower alpha (explore more)
    if num_words <= 3:
        alpha = min(base_alpha + 0.10, 0.30)
    elif num_words <= 6:
        alpha = base_alpha
    else:
        alpha = max(base_alpha - 0.05, 0.08)

    # More seeds → slightly lower alpha (more to explore from)
    if num_seeds > 5:
        alpha = max(alpha - 0.03, 0.05)

    # Adaptive max_iter
    # Complex queries might benefit from more iterations
    if num_words > 6:
        max_iter = min(base_max_iter + 1, 5)
    else:
        max_iter = base_max_iter

    # Adaptive frontier sizing
    # Small corpus → smaller frontier (less to explore)
    # Large corpus → larger frontier
    if corpus_size < 100:
        max_nodes = min(base_max_nodes // 10, 5000)
        max_edges = min(base_max_edges // 10, 25000)
    elif corpus_size < 1000:
        max_nodes = min(base_max_nodes // 2, 25000)
        max_edges = min(base_max_edges // 2, 125000)
    elif corpus_size < 10000:
        max_nodes = base_max_nodes
        max_edges = base_max_edges
    else:
        max_nodes = min(base_max_nodes * 2, 100000)
        max_edges = min(base_max_edges * 2, 500000)

    # More seeds → larger frontier (need more exploration)
    if num_seeds > 10:
        max_nodes = int(max_nodes * 1.2)
        max_edges = int(max_edges * 1.2)

    # Adaptive max_hops
    # Complex queries with many seeds → explore more hops
    if num_words > 6 and num_seeds > 5:
        max_hops = min(base_max_hops + 1, 3)
    else:
        max_hops = base_max_hops

    return AdaptiveParams(
        alpha=round(alpha, 3),
        max_iter=max_iter,
        max_nodes=min(max_nodes, 100000),
        max_edges=min(max_edges, 500000),
        max_hops=max_hops,
    )
