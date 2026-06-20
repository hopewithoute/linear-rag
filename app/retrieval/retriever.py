from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time
import logging

from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.reranker import rerank_contexts, RerankerConfig
from app.graph.frontier import expand_frontier
from app.graph.ppr import run_ppr
from app.graph.bfs import bfs_with_rerank
from app.graph.adaptive import compute_adaptive_params
from app.graph.snapshot import get_active_snapshot, GraphSnapshot, load_snapshot, set_active_snapshot
from app.storage.search_repository import SearchRepository
from app.storage.models import GraphSnapshot as GraphSnapshotModel
from sqlalchemy import select as sel
from app.settings import get_settings

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    query: str
    retrieval_mode: str
    snapshot_id: Optional[int]
    contexts: List[Dict[str, Any]]
    debug: Dict[str, Any]


async def _load_fallback_snapshot(repo: SearchRepository, settings: Any) -> Optional[GraphSnapshot]:
    try:
        stmt = sel(GraphSnapshotModel).where(GraphSnapshotModel.active == True).order_by(GraphSnapshotModel.snapshot_id.desc()).limit(1)
        result = await repo.session.execute(stmt)
        entry = result.scalar_one_or_none()
        if entry:
            gs = load_snapshot(entry.snapshot_id, settings.SNAPSHOT_DIR)
            if gs:
                set_active_snapshot(gs)
                return gs
    except Exception as e:
        logger.warning(f"Failed to load fallback snapshot: {e}")
    return None


async def _build_bipartite_mapping(repo: SearchRepository, seed_chunk_ids: set) -> tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    chunk_to_entities_map, entity_to_chunks_map = {}, {}
    if not seed_chunk_ids:
        return chunk_to_entities_map, entity_to_chunks_map

    ents = await repo.get_mentions_for_chunks(list(seed_chunk_ids))
    for cid, e_list in ents.items():
        for e in e_list:
            chunk_to_entities_map.setdefault(cid, []).append(e)
            entity_to_chunks_map.setdefault(e, []).append(cid)

    if chunk_to_entities_map:
        all_ents = list({e for el in chunk_to_entities_map.values() for e in el})
        ent_chunks = await repo.get_chunks_for_entities(all_ents)
        for e, cs in ent_chunks.items():
            entity_to_chunks_map.setdefault(e, [])
            for c in cs:
                if c not in seed_chunk_ids:
                    entity_to_chunks_map[e].append(c)

    return chunk_to_entities_map, entity_to_chunks_map


async def _run_hybrid_bfs(
    repo: SearchRepository,
    query: str,
    seed_scores: Dict[int, float],
    top_k: int,
    settings: Any
) -> tuple[List[int], Dict[int, float], Dict[int, List[str]]]:
    seed_chunk_ids = set(seed_scores.keys())
    chunk_to_entities_map, entity_to_chunks_map = await _build_bipartite_mapping(repo, seed_chunk_ids)

    all_chunk_ids = list(set(list(seed_chunk_ids) + [c for cs in entity_to_chunks_map.values() for c in cs]))
    chunk_data = await repo.get_chunks_by_ids(all_chunk_ids)
    chunk_texts = {cid: data.get("text", "") for cid, data in chunk_data.items()}

    bfs_result = bfs_with_rerank(
        seed_scores=seed_scores,
        chunk_to_entities=chunk_to_entities_map,
        entity_to_chunks=entity_to_chunks_map,
        query=query,
        chunk_texts=chunk_texts,
        max_iterations=3,
        iteration_threshold=0.4,
        max_chunks=top_k * 2,
        max_entity_degree=settings.MAX_ENTITY_DEGREE_PER_QUERY,
    )

    evidence_paths = {cid: bfs_result.evidence_paths[cid] for cid in bfs_result.chunk_ids if cid in bfs_result.evidence_paths}
    ranked = sorted(bfs_result.scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    return [cid for cid, _ in ranked], {cid: s for cid, s in ranked}, evidence_paths


async def _run_hybrid_ppr(
    repo: SearchRepository,
    query: str,
    seed_scores: Dict[int, float],
    snapshot: GraphSnapshot,
    top_k: int,
    settings: Any,
    debug_stats: Dict[str, int]
) -> tuple[List[int], Dict[int, float], Dict[int, List[str]]]:
    seed_chunk_ids = set(seed_scores.keys())
    chunk_to_entities_map, entity_to_chunks_map = await _build_bipartite_mapping(repo, seed_chunk_ids)

    mapped_seed = {snapshot.chunk_id_map[cid]: score for cid, score in seed_scores.items() if cid in snapshot.chunk_id_map}

    adaptive = compute_adaptive_params(
        query=query,
        seed_scores=mapped_seed,
        corpus_size=snapshot.chunk_count,
        base_alpha=settings.PPR_ALPHA,
        base_max_iter=settings.PPR_MAX_ITER,
        base_max_nodes=settings.PPR_MAX_FRONTIER_NODES,
        base_max_edges=settings.PPR_MAX_FRONTIER_EDGES,
        base_max_hops=settings.PPR_MAX_HOPS,
    )

    frontier = expand_frontier(
        seed_chunk_ids=set(mapped_seed.keys()),
        entity_to_chunks={e: [snapshot.chunk_id_map.get(c, c) for c in cs] for e, cs in entity_to_chunks_map.items()},
        chunk_to_entities={snapshot.chunk_id_map.get(c, c): es for c, es in chunk_to_entities_map.items() if c in snapshot.chunk_id_map},
        max_nodes=adaptive.max_nodes,
        max_edges=adaptive.max_edges,
        max_hops=adaptive.max_hops,
        max_entity_degree=settings.MAX_ENTITY_DEGREE_PER_QUERY,
    )

    debug_stats["frontier_nodes"] = len(frontier.chunk_ids) + len(frontier.entity_ids)
    debug_stats["frontier_edges"] = frontier.num_edges
    debug_stats["ppr_iter"] = settings.PPR_MAX_ITER

    chunk_nodes = frontier.chunk_ids
    p_seed = {nid: score for nid, score in mapped_seed.items() if nid in frontier.node_ids}
    if not p_seed:
        chunk_frontier = [n for n in frontier.node_ids if n < snapshot.chunk_count]
        p_seed = {nid: 1.0 / len(chunk_frontier) for nid in chunk_frontier} if chunk_frontier else {}

    ppr_scores, ppr_paths = run_ppr(
        matrix=snapshot.matrix,
        seed_scores=p_seed,
        chunk_nodes=chunk_nodes,
        active_mask=snapshot.active_mask,
        alpha=adaptive.alpha,
        max_iter=adaptive.max_iter,
    )

    reverse_map = {v: k for k, v in snapshot.chunk_id_map.items()}
    evidence_paths = {}
    
    for matrix_idx, parents in ppr_paths.items():
        orig_cid = reverse_map.get(matrix_idx, matrix_idx)
        path = [f"query -> chunk_{orig_cid}"]
        for parent_idx in parents[:3]:
            orig_parent = reverse_map.get(parent_idx, parent_idx)
            if orig_parent != orig_cid:
                if parent_idx >= snapshot.chunk_count:
                    path.append(f"entity_{parent_idx - snapshot.chunk_count}")
                else:
                    path.append(f"chunk_{orig_parent}")
        evidence_paths[orig_cid] = path

    final_scores = {reverse_map.get(matrix_idx, matrix_idx): ppr_s for matrix_idx, ppr_s in ppr_scores.items()}
    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    return [cid for cid, _ in ranked], {cid: s for cid, s in ranked}, evidence_paths


async def retrieve(
    repo: SearchRepository,
    query: str,
    top_k: int = 12,
    retrieval_mode: str = "hybrid_ppr",
    query_embedding: Optional[List[float]] = None,
) -> RetrievalResult:
    settings = get_settings()
    timings: Dict[str, float] = {}
    debug_stats: Dict[str, Any] = {"frontier_nodes": 0, "frontier_edges": 0, "ppr_iter": 0}

    snapshot = get_active_snapshot()
    if snapshot is None:
        snapshot = await _load_fallback_snapshot(repo, settings)

    dense_hits, bm25_hits = [], []

    if retrieval_mode in ("hybrid_ppr", "hybrid_bfs", "hybrid", "vector_only"):
        t0 = time.monotonic()
        dense_hits = await repo.vector_search(query_embedding, top_k=settings.VECTOR_TOP_K) if query_embedding else []
        timings["vector"] = round((time.monotonic() - t0) * 1000)

    if retrieval_mode in ("hybrid_ppr", "hybrid_bfs", "hybrid", "bm25_only"):
        t0 = time.monotonic()
        bm25_hits = await repo.bm25_search(query, top_k=settings.BM25_TOP_K)
        timings["bm25"] = round((time.monotonic() - t0) * 1000)

    if retrieval_mode == "vector_only":
        seed_scores = {h["chunk_id"]: h["score"] for h in dense_hits[:top_k]}
    elif retrieval_mode == "bm25_only":
        seed_scores = {h["chunk_id"]: h["score"] for h in bm25_hits[:top_k]}
    else:
        seed_scores = reciprocal_rank_fusion(
            dense_hits, bm25_hits,
            w_dense=settings.DENSE_WEIGHT, w_bm25=settings.BM25_WEIGHT, k=settings.RRF_K,
        )

    seed_chunk_ids = set(seed_scores.keys())
    evidence_paths: Dict[int, List[str]] = {}

    if retrieval_mode == "hybrid_bfs" and seed_chunk_ids:
        t0 = time.monotonic()
        top_chunk_ids, chunk_scores, evidence_paths = await _run_hybrid_bfs(repo, query, seed_scores, top_k, settings)
        timings["bfs"] = round((time.monotonic() - t0) * 1000)

    elif snapshot is not None and retrieval_mode == "hybrid_ppr" and seed_chunk_ids and snapshot.chunk_id_map:
        t0 = time.monotonic()
        top_chunk_ids, chunk_scores, evidence_paths = await _run_hybrid_ppr(repo, query, seed_scores, snapshot, top_k, settings, debug_stats)
        timings["ppr"] = round((time.monotonic() - t0) * 1000)
    else:
        ranked = sorted(seed_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_chunk_ids = [cid for cid, _ in ranked]
        chunk_scores = {cid: s for cid, s in ranked}

    chunk_data = await repo.get_chunks_by_ids(top_chunk_ids)

    contexts = []
    for cid in top_chunk_ids:
        data = chunk_data.get(cid, {})
        if not data:
            continue
        
        path = evidence_paths.get(cid, [f"query -> chunk_{cid}"])
        contexts.append({
            "chunk_id": cid,
            "document_id": data.get("document_id"),
            "score": round(chunk_scores.get(cid, 0), 4),
            "text": data.get("text", ""),
            "evidence_path": path,
            "rerank_boost": None,
        })

    if contexts and retrieval_mode in ("hybrid_ppr", "hybrid"):
        t0 = time.monotonic()
        reranker_cfg = RerankerConfig(
            mode=settings.RERANKER_MODE,
            cross_encoder_model=settings.RERANKER_CROSS_ENCODER_MODEL,
            boost_exact_match=settings.RERANKER_BOOST_EXACT_MATCH,
            boost_short_chunk=settings.RERANKER_BOOST_SHORT_CHUNK,
            penalize_long_chunk=settings.RERANKER_PENALIZE_LONG_CHUNK,
        )
        contexts = rerank_contexts(contexts, query, reranker_cfg)
        timings["rerank"] = round((time.monotonic() - t0) * 1000)

    timings["total"] = sum(timings.values())

    debug_stats.update({
        "vector_hits": len(dense_hits),
        "bm25_hits": len(bm25_hits),
        "latency_ms": timings,
    })

    return RetrievalResult(
        query=query,
        retrieval_mode=retrieval_mode,
        snapshot_id=snapshot.snapshot_id if snapshot else None,
        contexts=contexts,
        debug=debug_stats,
    )
