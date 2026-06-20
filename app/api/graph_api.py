from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Tuple, Dict, Any
from app.api.auth import verify_api_key
from app.storage.db import get_db_session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete
from app.storage.models import Chunk, Entity, Mention, GraphSnapshot, Document
from app.graph.coo_log import COOLog
from app.graph.compiler import compile_snapshot
from app.graph.snapshot import GraphSnapshot as GSSnapshot, save_snapshot, set_active_snapshot, get_active_snapshot
from app.storage.search_repository import SearchRepository
from app.settings import get_settings
import numpy as np
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class CompileRequest(BaseModel):
    mode: str = "full"
    publish_on_success: bool = True


class CompileResponse(BaseModel):
    status: str
    snapshot_id: Optional[int] = None
    num_chunk_nodes: int = 0
    num_entity_nodes: int = 0
    num_directed_edges: int = 0
    tombstones_removed: int = 0


class SnapshotInfo(BaseModel):
    snapshot_id: int
    version: str
    num_chunk_nodes: int
    num_entity_nodes: int
    num_directed_edges: int
    active: bool


async def _check_and_run_compaction(session: AsyncSession, mode: str, settings: Any) -> int:
    """Check tombstone ratio and run auto-compaction if necessary."""
    if mode not in ("full", "compact"):
        return 0

    inactive_count = await session.execute(select(func.count(Chunk.chunk_id)).where(Chunk.active == False))
    inactive = inactive_count.scalar() or 0
    
    if inactive == 0 and mode != "compact":
        return 0

    total_all = await session.execute(select(func.count(Chunk.chunk_id)))
    total_all_count = total_all.scalar() or 1

    tombstone_ratio = inactive / total_all_count
    if tombstone_ratio > settings.TOMBSTONE_COMPACTION_THRESHOLD or mode == "compact":
        return await _compact_mentions(session, inactive)
    return 0


async def _build_coo_log_and_maps(session: AsyncSession) -> Tuple[Optional[COOLog], Dict[int, int], Dict[int, int], List[int]]:
    """Fetch active mentions and build COO log alongside entity/chunk ID mappings."""
    mentions_stmt = (
        select(Mention.chunk_id, Mention.entity_id, Mention.confidence)
        .join(Chunk, Chunk.chunk_id == Mention.chunk_id)
        .where(Chunk.active == True)
    )
    result = await session.execute(mentions_stmt)
    rows = result.all()

    if not rows:
        return None, {}, {}, []

    all_chunk_ids = sorted({row.chunk_id for row in rows})
    chunk_id_map = {cid: idx for idx, cid in enumerate(all_chunk_ids)}

    all_entity_ids = sorted({row.entity_id for row in rows})
    entity_id_map = {eid: idx for idx, eid in enumerate(all_entity_ids)}

    coo = COOLog()
    for row in rows:
        chunk_node = chunk_id_map[row.chunk_id]
        entity_node = len(all_chunk_ids) + entity_id_map[row.entity_id]
        coo.append_bidirectional(chunk_node, entity_node, row.confidence)

    return coo, chunk_id_map, entity_id_map, all_entity_ids


async def _save_and_activate_snapshot(
    session: AsyncSession, 
    matrix: Any, 
    chunk_id_map: Dict[int, int], 
    entity_count: int, 
    coo_length: int, 
    active_mask: np.ndarray, 
    publish_on_success: bool, 
    settings: Any
) -> int:
    """Save the snapshot metadata to DB, matrix to disk, and optionally activate."""
    chunk_count = len(chunk_id_map)
    snapshot_entry = GraphSnapshot(
        version=f"v{chunk_count}_{entity_count}",
        num_chunk_nodes=chunk_count,
        num_entity_nodes=entity_count,
        num_directed_edges=coo_length,
        format="csr",
        active=False,
    )
    session.add(snapshot_entry)
    await session.flush()

    gs = GSSnapshot(
        snapshot_id=snapshot_entry.snapshot_id,
        version=snapshot_entry.version,
        matrix=matrix,
        chunk_count=chunk_count,
        entity_count=entity_count,
        active_mask=active_mask,
        chunk_id_map=chunk_id_map,
    )

    save_snapshot(gs, settings.SNAPSHOT_DIR)

    if publish_on_success:
        prev = await session.execute(select(GraphSnapshot).where(GraphSnapshot.active == True))
        for old in prev.scalars().all():
            old.active = False

        snapshot_entry.active = True
        set_active_snapshot(gs)

    return snapshot_entry.snapshot_id


@router.post("/graph/compile", response_model=CompileResponse)
async def compile_graph(
    request: CompileRequest,
    _: None = Depends(verify_api_key),
    session: AsyncSession = Depends(get_db_session),
):
    settings = get_settings()

    try:
        repo = SearchRepository(session)
        total_chunks = await repo.get_total_chunks()
        
        if total_chunks == 0:
            return CompileResponse(status="empty")

        tombstones_removed = await _check_and_run_compaction(session, request.mode, settings)
        
        coo, chunk_id_map, entity_id_map, all_entity_ids = await _build_coo_log_and_maps(session)
        if not coo:
            return CompileResponse(status="empty")

        entity_doc_freq = await repo.get_entity_doc_freq()
        entity_degree = await repo.get_entity_degree()
        active_mask = np.ones(len(chunk_id_map), dtype=bool)

        entity_names = {}
        for i in range(0, len(all_entity_ids), 30000):
            batch_ids = all_entity_ids[i:i + 30000]
            stmt = select(Entity.entity_id, Entity.canonical_name).where(Entity.entity_id.in_(batch_ids))
            for row in (await session.execute(stmt)).all():
                entity_names[row.entity_id] = row.canonical_name

        matrix = compile_snapshot(
            coo_log=coo,
            chunk_count=len(chunk_id_map),
            entity_count=len(all_entity_ids),
            entity_doc_freq=entity_doc_freq,
            entity_degree=entity_degree,
            active_chunks_mask=active_mask,
            entity_names=entity_names,
        )

        snapshot_id = await _save_and_activate_snapshot(
            session, matrix, chunk_id_map, len(all_entity_ids), len(coo), active_mask, request.publish_on_success, settings
        )

        await session.commit()

        return CompileResponse(
            status="compiled",
            snapshot_id=snapshot_id,
            num_chunk_nodes=len(chunk_id_map),
            num_entity_nodes=len(all_entity_ids),
            num_directed_edges=len(coo),
            tombstones_removed=tombstones_removed,
        )

    except Exception as e:
        await session.rollback()
        logger.exception(f"Graph compile failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _compact_mentions(session: AsyncSession, inactive_count: int) -> int:
    """Remove mentions for inactive chunks and update entity stats."""
    delete_stmt = delete(Mention).where(
        Mention.chunk_id.in_(
            select(Chunk.chunk_id).where(Chunk.active == False)
        )
    )
    result = await session.execute(delete_stmt)
    deleted = result.rowcount

    repo = SearchRepository(session)
    await repo.update_entity_stats()

    logger.info(f"Compaction: removed {deleted} mentions for {inactive_count} inactive chunks")
    return deleted


@router.get("/graph/snapshot/active", response_model=Optional[SnapshotInfo])
async def get_active_snapshot_info(
    _: None = Depends(verify_api_key),
    session: AsyncSession = Depends(get_db_session),
):
    gs = get_active_snapshot()
    if gs is None:
        stmt = select(GraphSnapshot).where(GraphSnapshot.active == True).order_by(GraphSnapshot.snapshot_id.desc()).limit(1)
        result = await session.execute(stmt)
        entry = result.scalar_one_or_none()
        if entry is None:
            return None
        return SnapshotInfo(
            snapshot_id=entry.snapshot_id,
            version=entry.version,
            num_chunk_nodes=entry.num_chunk_nodes,
            num_entity_nodes=entry.num_entity_nodes,
            num_directed_edges=entry.num_directed_edges,
            active=entry.active,
        )

    return SnapshotInfo(
        snapshot_id=gs.snapshot_id,
        version=gs.version,
        num_chunk_nodes=gs.chunk_count,
        num_entity_nodes=gs.entity_count,
        num_directed_edges=0,
        active=True,
    )
