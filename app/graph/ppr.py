import numpy as np
from scipy import sparse
from typing import Dict, Set, Optional, List, Tuple


def run_ppr(
    matrix: sparse.csr_matrix,
    seed_scores: Dict[int, float],
    chunk_nodes: Set[int],
    active_mask: Optional[np.ndarray] = None,
    alpha: float = 0.15,
    max_iter: int = 3,
) -> Tuple[Dict[int, float], Dict[int, List[int]]]:
    """Run PPR and track paths through entities.
    
    Returns:
        scores: {matrix_idx: score}
        paths: {matrix_idx: [path_nodes]} - trace of score propagation
    """
    num_nodes = matrix.shape[0]
    r = np.zeros(num_nodes, dtype=np.float64)
    p = np.zeros(num_nodes, dtype=np.float64)

    # Track which nodes received score from which nodes
    parent_map: Dict[int, List[int]] = {nid: [] for nid in seed_scores}

    total = sum(seed_scores.values())
    if total <= 0:
        return {}, {}

    for node_id, score in seed_scores.items():
        if node_id < num_nodes:
            p[node_id] = score / total
            r[node_id] = p[node_id]

    for _ in range(max_iter):
        if active_mask is not None:
            for node in chunk_nodes:
                if node < num_nodes and not active_mask[node]:
                    r[node] = 0.0

        r_next = (1 - alpha) * matrix.T.dot(r) + alpha * p

        # Track score propagation using sparse matrix structure
        # For nodes that gained significant score, find their sources
        diff = r_next - r
        significant_nodes = np.where(diff > 0.001)[0]

        for row_idx in significant_nodes:
            # Get non-zero entries in this row (sources that contributed)
            row_start = matrix.indptr[row_idx]
            row_end = matrix.indptr[row_idx + 1]
            
            for idx in range(row_start, row_end):
                col_idx = matrix.indices[idx]
                if r[col_idx] > 0:  # source had score
                    if row_idx not in parent_map:
                        parent_map[row_idx] = []
                    parent_map[row_idx].append(col_idx)

        r = r_next

    if active_mask is not None:
        for node in chunk_nodes:
            if node < num_nodes and not active_mask[node]:
                r[node] = 0.0

    scores: Dict[int, float] = {}
    for node in chunk_nodes:
        if node < num_nodes and r[node] > 0:
            scores[node] = float(r[node])

    return scores, parent_map
