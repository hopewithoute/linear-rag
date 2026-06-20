import math
import numpy as np
from scipy import sparse
from typing import Dict, List, Tuple, Optional
from app.graph.coo_log import COOLog
from app.graph.stop_entities import get_entity_penalty


def compute_entity_idf(entity_doc_freq: Dict[int, int], total_chunks: int) -> Dict[int, float]:
    idf = {}
    for entity_id, df in entity_doc_freq.items():
        idf[entity_id] = math.log((total_chunks + 1) / (df + 1))
    return idf


def apply_hub_penalty(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    entity_idf: Dict[int, float],
    entity_degree: Dict[int, int],
    chunk_count: int,
    entity_names: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    adjusted = values.copy()
    for i in range(len(rows)):
        node = rows[i]
        if node >= chunk_count:
            entity_id = node - chunk_count
            idf = entity_idf.get(entity_id, 1.0)
            deg = entity_degree.get(entity_id, 1)
            penalty = idf / math.sqrt(max(deg, 1))
            
            # Apply stop-entity penalty
            if entity_names:
                entity_name = entity_names.get(entity_id, "")
                stop_penalty = get_entity_penalty(entity_name)
                penalty *= stop_penalty
            
            adjusted[i] *= penalty
    return adjusted


def normalize_columns(rows: np.ndarray, cols: np.ndarray, values: np.ndarray, num_nodes: int) -> np.ndarray:
    col_sums = np.zeros(num_nodes, dtype=np.float64)
    for i in range(len(cols)):
        col_sums[cols[i]] += values[i]

    normalized = values.copy()
    for i in range(len(cols)):
        s = col_sums[cols[i]]
        if s > 0:
            normalized[i] /= s
    return normalized


def compile_snapshot(
    coo_log: COOLog,
    chunk_count: int,
    entity_count: int,
    entity_doc_freq: Dict[int, int],
    entity_degree: Dict[int, int],
    active_chunks_mask: Optional[np.ndarray] = None,
    entity_names: Optional[Dict[int, str]] = None,
) -> sparse.csr_matrix:
    if len(coo_log) == 0:
        return sparse.csr_matrix((chunk_count + entity_count, chunk_count + entity_count), dtype=np.float32)

    rows, cols, values = coo_log.to_arrays()

    if active_chunks_mask is not None:
        mask = np.ones(len(rows), dtype=bool)
        for i in range(len(rows)):
            node = rows[i]
            if node < chunk_count and not active_chunks_mask[node]:
                mask[i] = False
            src = cols[i]
            if src < chunk_count and not active_chunks_mask[src]:
                mask[i] = False
        rows, cols, values = rows[mask], cols[mask], values[mask]

    entity_idf = compute_entity_idf(entity_doc_freq, chunk_count)
    adjusted = apply_hub_penalty(rows, cols, values, entity_idf, entity_degree, chunk_count, entity_names)

    num_nodes = chunk_count + entity_count
    normalized = normalize_columns(rows, cols, adjusted, num_nodes)

    matrix = sparse.coo_matrix(
        (normalized, (rows, cols)),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    )
    return matrix.tocsr()
