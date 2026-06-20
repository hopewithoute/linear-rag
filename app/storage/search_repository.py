from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from app.storage.models import Chunk, Document, Entity, Mention
from app.storage.tenant import normalize_tenant_id


class SearchRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def vector_search(
        self, query_embedding: List[float], top_k: int = 50, tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        distance_func = Chunk.embedding.cosine_distance(query_embedding)
        stmt = (
            select(Chunk.chunk_id, Chunk.document_id, Chunk.text, distance_func.label("distance"))
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.active == True)
            .order_by(distance_func)
            .limit(top_k)
        )
        result = await self.session.execute(stmt)
        return [
            {"chunk_id": row.chunk_id, "document_id": row.document_id, "text": row.text, "score": 1.0 - row.distance}
            for row in result.all()
        ]

    async def bm25_search(self, query: str, top_k: int = 50, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        tsquery = func.websearch_to_tsquery("simple", query)
        rank_func = func.ts_rank_cd(Chunk.search_vector, tsquery)
        stmt = (
            select(Chunk.chunk_id, Chunk.document_id, Chunk.text, rank_func.label("rank"))
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.active == True)
            .where(Chunk.search_vector.op("@@")(tsquery))
            .order_by(rank_func.desc())
            .limit(top_k)
        )
        result = await self.session.execute(stmt)
        return [
            {"chunk_id": row.chunk_id, "document_id": row.document_id, "text": row.text, "score": row.rank}
            for row in result.all()
        ]

    async def get_chunks_by_ids(self, chunk_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        if not chunk_ids:
            return {}
        stmt = (
            select(Chunk.chunk_id, Chunk.document_id, Chunk.text)
            .where(Chunk.chunk_id.in_(chunk_ids))
        )
        result = await self.session.execute(stmt)
        return {row.chunk_id: {"chunk_id": row.chunk_id, "document_id": row.document_id, "text": row.text} for row in result.all()}

    async def get_chunks_by_document_ids(self, document_ids: List[int]) -> Dict[int, List[Dict[str, Any]]]:
        if not document_ids:
            return {}
        stmt = (
            select(Chunk.chunk_id, Chunk.document_id, Chunk.text)
            .where(Chunk.document_id.in_(document_ids))
            .order_by(Chunk.document_id, Chunk.chunk_id)
        )
        result = await self.session.execute(stmt)
        mapping: Dict[int, List[Dict[str, Any]]] = {}
        for row in result.all():
            mapping.setdefault(row.document_id, []).append({
                "chunk_id": row.chunk_id,
                "document_id": row.document_id,
                "text": row.text
            })
        return mapping

    async def get_mentions_for_chunks(self, chunk_ids: List[int]) -> Dict[int, List[int]]:
        if not chunk_ids:
            return {}
        stmt = select(Mention.chunk_id, Mention.entity_id).where(Mention.chunk_id.in_(chunk_ids))
        result = await self.session.execute(stmt)
        mapping: Dict[int, List[int]] = {}
        for row in result.all():
            mapping.setdefault(row.chunk_id, []).append(row.entity_id)
        return mapping

    async def get_chunks_for_entities(self, entity_ids: List[int]) -> Dict[int, List[int]]:
        if not entity_ids:
            return {}
        stmt = select(Mention.entity_id, Mention.chunk_id).where(Mention.entity_id.in_(entity_ids))
        result = await self.session.execute(stmt)
        mapping: Dict[int, List[int]] = {}
        for row in result.all():
            mapping.setdefault(row.entity_id, []).append(row.chunk_id)
        return mapping

    async def get_active_chunks_mask(self, total_chunks: int) -> "np.ndarray":
        import numpy as np
        from sqlalchemy import select as sel
        stmt = sel(Chunk.chunk_id).where(Chunk.active == True)
        result = await self.session.execute(stmt)
        active_ids = {row[0] for row in result.all()}

        all_stmt = sel(Chunk.chunk_id)
        all_result = await self.session.execute(all_stmt)
        all_ids = sorted([row[0] for row in all_result.all()])
        id_to_idx = {cid: idx for idx, cid in enumerate(all_ids)}

        mask = np.zeros(total_chunks, dtype=bool)
        for cid in active_ids:
            idx = id_to_idx.get(cid)
            if idx is not None and idx < total_chunks:
                mask[idx] = True
        return mask

    async def get_chunk_embeddings(self, chunk_ids: List[int]) -> Dict[int, List[float]]:
        if not chunk_ids:
            return {}
        stmt = select(Chunk.chunk_id, Chunk.embedding).where(Chunk.chunk_id.in_(chunk_ids))
        result = await self.session.execute(stmt)
        return {row.chunk_id: row.embedding.to_list() if hasattr(row.embedding, "to_list") else list(row.embedding) for row in result.all()}

    async def get_entity_doc_freq(self) -> Dict[int, int]:
        from sqlalchemy import func as sqlfunc
        stmt = (
            select(Mention.entity_id, sqlfunc.count(sqlfunc.distinct(Mention.chunk_id)).label("df"))
            .group_by(Mention.entity_id)
        )
        result = await self.session.execute(stmt)
        return {row.entity_id: row.df for row in result.all()}

    async def get_entity_degree(self) -> Dict[int, int]:
        from sqlalchemy import func as sqlfunc
        stmt = (
            select(Mention.entity_id, sqlfunc.count(Mention.mention_id).label("degree"))
            .group_by(Mention.entity_id)
        )
        result = await self.session.execute(stmt)
        return {row.entity_id: row.degree for row in result.all()}

    async def update_entity_stats(self, entity_ids: Optional[List[int]] = None) -> None:
        """Update degree and document_frequency for entities based on current mentions."""
        from sqlalchemy import select as sel, func as sqlfunc
        
        df_stmt = sel(Mention.entity_id, sqlfunc.count(sqlfunc.distinct(Mention.chunk_id))).group_by(Mention.entity_id)
        if entity_ids:
            df_stmt = df_stmt.where(Mention.entity_id.in_(entity_ids))
        df_result = await self.session.execute(df_stmt)
        df_map = {row[0]: row[1] for row in df_result.all()}

        degree_stmt = sel(Mention.entity_id, sqlfunc.count(Mention.mention_id)).group_by(Mention.entity_id)
        if entity_ids:
            degree_stmt = degree_stmt.where(Mention.entity_id.in_(entity_ids))
        degree_result = await self.session.execute(degree_stmt)
        degree_map = {row[0]: row[1] for row in degree_result.all()}

        ent_stmt = sel(Entity)
        if entity_ids:
            ent_stmt = ent_stmt.where(Entity.entity_id.in_(entity_ids))
        
        entities = await self.session.execute(ent_stmt)
        for entity in entities.scalars().all():
            entity.document_frequency = df_map.get(entity.entity_id, 0)
            entity.degree = degree_map.get(entity.entity_id, 0)

    async def get_total_chunks(self) -> int:
        from sqlalchemy import func as sqlfunc
        stmt = select(sqlfunc.count(Chunk.chunk_id)).where(Chunk.active == True)
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def get_total_entities(self) -> int:
        from sqlalchemy import func as sqlfunc
        stmt = select(sqlfunc.count(Entity.entity_id)).where(Entity.active == True)
        result = await self.session.execute(stmt)
        return result.scalar() or 0
