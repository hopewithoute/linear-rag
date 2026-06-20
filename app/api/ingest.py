from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import logging

from app.api.auth import verify_api_key
from app.storage.db import get_db_session
from app.storage.models import Document, Chunk, Mention, Entity, EntityAlias
from app.chunking.factory import create_chunker
from app.chunking.contracts import ChunkDocumentInput, ChunkingOptions
from app.extraction.factory import create_term_extractor, get_gliner_labels
from app.extraction.contracts import Chunk as ExtractionChunk
from app.extraction.canonicalizer import normalize_entity, fuzzy_match_entity
from app.embedding.factory import create_embedding_runtime
from app.settings import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)

class IngestRequest(BaseModel):
    document_id: Optional[str] = None
    source_uri: Optional[str] = None
    text: str
    metadata: Optional[Dict[str, Any]] = None
    tenant_id: Optional[str] = None

class IngestResponse(BaseModel):
    status: str
    document_id: str
    chunks_created: int
    entities_extracted: int
    edges_created: int

async def _resolve_entities(session: AsyncSession, entities: List[Any], settings: Any) -> Dict[str, int]:
    entity_map: Dict[str, int] = {}
    requested_norms = []
    for ent in entities:
        norm = normalize_entity(ent.text)
        if norm not in requested_norms:
            requested_norms.append(norm)
        if len(requested_norms) >= settings.MAX_ENTITIES_PER_CHUNK:
            break

    # 1. Bulk exact match
    exact_matches = await session.execute(
        select(Entity).where(Entity.canonical_name.in_(requested_norms), Entity.active == True)
    )
    for existing_entity in exact_matches.scalars().all():
        entity_map[existing_entity.canonical_name] = existing_entity.entity_id

    # 2. Fuzzy match & Create new
    remaining_norms = [norm for norm in requested_norms if norm not in entity_map]
    if remaining_norms:
        existing_name_map = {}
        for norm in remaining_norms:
            prefix = norm[:3]
            candidates = await session.execute(
                select(Entity.canonical_name, Entity.entity_id)
                .where(Entity.canonical_name.ilike(f"{prefix}%"), Entity.active == True)
                .limit(100)
            )
            for row in candidates.all():
                existing_name_map[row[0]] = row[1]
                
        existing_names = list(existing_name_map.keys())

        existing_aliases_result = await session.execute(
            select(EntityAlias.alias).where(EntityAlias.alias.in_(remaining_norms))
        )
        existing_aliases_set = {row[0] for row in existing_aliases_result.all()}

        for norm in remaining_norms:
            fuzzy_match = fuzzy_match_entity(norm, existing_names, threshold=0.85)
            if fuzzy_match:
                matched_id = existing_name_map[fuzzy_match]
                entity_map[norm] = matched_id
                if norm not in existing_aliases_set:
                    session.add(EntityAlias(alias=norm, entity_id=matched_id, confidence=0.9))
            else:
                ent_label = next(e.label for e in entities if normalize_entity(e.text) == norm)
                new_entity = Entity(canonical_name=norm, entity_type=ent_label)
                session.add(new_entity)
                await session.flush()
                entity_map[norm] = new_entity.entity_id
                existing_names.append(norm)
                existing_name_map[norm] = new_entity.entity_id

    return entity_map

async def _create_mentions(session: AsyncSession, entities: List[Any], entity_map: Dict[str, int], chunk_ids: List[int]) -> int:
    edges_created = 0
    mentions_to_check = []
    for ent in entities:
        norm = normalize_entity(ent.text)
        if norm not in entity_map:
            continue
        eid = entity_map[norm]
        cid_idx = int(ent.chunk_id)
        if cid_idx < len(chunk_ids):
            cid = chunk_ids[cid_idx]
            mentions_to_check.append((cid, eid, ent))

    if mentions_to_check:
        cids = list({m[0] for m in mentions_to_check})
        eids = list({m[1] for m in mentions_to_check})
        existing_mentions_result = await session.execute(
            select(Mention.chunk_id, Mention.entity_id).where(
                Mention.chunk_id.in_(cids),
                Mention.entity_id.in_(eids)
            )
        )
        existing_mentions = {(row.chunk_id, row.entity_id) for row in existing_mentions_result.all()}

        for cid, eid, ent in mentions_to_check:
            if (cid, eid) not in existing_mentions:
                session.add(Mention(chunk_id=cid, entity_id=eid, surface_text=ent.text, confidence=ent.confidence))
                existing_mentions.add((cid, eid))
                edges_created += 1

    return edges_created


@router.post("/ingest/document", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    _: None = Depends(verify_api_key),
    session: AsyncSession = Depends(get_db_session),
):
    settings = get_settings()

    try:
        doc = Document(tenant_id=request.tenant_id or "", source_uri=request.source_uri, metadata_=request.metadata)
        session.add(doc)
        await session.flush()
        doc_id = doc.id

        chunker = create_chunker(settings)
        chunks = chunker.chunk_document(
            ChunkDocumentInput(content=request.text, source_type="text"),
            ChunkingOptions(target_tokens=settings.CHUNK_SIZE_TOKENS, overlap_tokens=settings.CHUNK_OVERLAP_TOKENS),
        )

        extractor = create_term_extractor(settings)
        ext_chunks = [ExtractionChunk(chunk_id=str(i), content=c.content) for i, c in enumerate(chunks)]
        labels = get_gliner_labels()
        entities = extractor.extract_terms(ext_chunks, labels, threshold=settings.TERM_EXTRACTOR_THRESHOLD)

        chunk_objs = []
        chunk_ids = []
        for c in chunks:
            chunk = Chunk(document_id=doc_id, text=c.content, start_offset=c.start_offset, end_offset=c.end_offset)
            session.add(chunk)
            await session.flush()
            chunk_objs.append(chunk)
            chunk_ids.append(chunk.chunk_id)

        try:
            embedder = create_embedding_runtime(settings)
            texts = [c.content for c in chunks]
            embeddings = []
            # Batch embedding requests to prevent 400 Bad Request from sidecar limit
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_emb = embedder.create_embeddings(batch_texts)
                embeddings.extend(batch_emb)
                
            for i, emb in enumerate(embeddings):
                if i < len(chunk_objs) and len(emb) == settings.EMBEDDING_DIMENSIONS:
                    chunk_objs[i].embedding = emb
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")

        entity_map = await _resolve_entities(session, entities, settings)
        edges_created = await _create_mentions(session, entities, entity_map, chunk_ids)
        
        if entity_map:
            from app.storage.search_repository import SearchRepository
            await SearchRepository(session).update_entity_stats(list(entity_map.values()))

        await session.commit()

        return IngestResponse(
            status="accepted",
            document_id=request.document_id or f"doc_{doc_id}",
            chunks_created=len(chunk_ids),
            entities_extracted=len(entity_map),
            edges_created=edges_created,
        )

    except Exception as e:
        await session.rollback()
        logger.exception(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
