from typing import List, Dict


def reciprocal_rank_fusion(
    dense_hits: List[Dict],
    bm25_hits: List[Dict],
    w_dense: float = 0.6,
    w_bm25: float = 0.4,
    k: int = 60,
) -> Dict[int, float]:
    scores: Dict[int, float] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        cid = hit["chunk_id"]
        scores[cid] = scores.get(cid, 0) + w_dense * 1.0 / (k + rank)

    for rank, hit in enumerate(bm25_hits, start=1):
        cid = hit["chunk_id"]
        scores[cid] = scores.get(cid, 0) + w_bm25 * 1.0 / (k + rank)

    return scores
