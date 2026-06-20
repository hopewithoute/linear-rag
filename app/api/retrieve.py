from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from app.api.auth import verify_api_key
from app.storage.db import get_db_session
from sqlalchemy.ext.asyncio import AsyncSession
from app.storage.search_repository import SearchRepository
from app.retrieval.linear_rag import LinearRagRetriever
from app.embedding.factory import create_embedding_runtime
from app.settings import get_settings
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class ContextItem(BaseModel):
    chunk_id: int
    document_id: Optional[int] = None
    score: float
    text: str


class RetrieveRequest(BaseModel):
    query: str
    tenant_id: Optional[str] = None
    top_k: int = Field(default=12, le=1000)
    mode: str = Field(default="hybrid")


class RetrieveResponse(BaseModel):
    query: str
    retrieval_mode: str
    contexts: List[ContextItem]
    debug: Dict[str, Any]


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_context(
    request: RetrieveRequest,
    _: None = Depends(verify_api_key),
    session: AsyncSession = Depends(get_db_session),
):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")

    settings = get_settings()

    try:
        repo = SearchRepository(session)
        try:
            embedder = create_embedding_runtime(settings)
        except Exception as e:
            logger.exception("Failed to initialize embedder")
            raise HTTPException(status_code=500, detail="Embedder initialization failed")

        if request.mode == "bm25_only":
            import time
            t0 = time.monotonic()
            hits = await repo.bm25_search(request.query, top_k=request.top_k, tenant_id=request.tenant_id)
            timings = {"total": round((time.monotonic() - t0) * 1000)}
            contexts = hits
            retrieval_mode = "bm25_only"
            debug = {"timings": timings}
        elif request.mode == "vector_only":
            import time
            t0 = time.monotonic()
            query_emb = embedder.create_embeddings([request.query])[0]
            hits = await repo.vector_search(query_emb, top_k=request.top_k, tenant_id=request.tenant_id)
            timings = {"total": round((time.monotonic() - t0) * 1000)}
            contexts = hits
            retrieval_mode = "vector_only"
            debug = {"timings": timings}
        else:
            retriever = LinearRagRetriever(repo=repo, embedding_model=embedder)
            result = await retriever.retrieve(
                query=request.query,
                top_k=request.top_k
            )
            contexts = result.contexts
            retrieval_mode = result.retrieval_mode
            debug = result.debug

        try:
            from app.storage.models import RetrievalAudit
            audit = RetrievalAudit(
                query=request.query,
                chunk_ids=[c["chunk_id"] for c in contexts],
                scores=[c["score"] for c in contexts],
                latency_ms=debug.get("timings", {}).get("total", 0)
            )
            session.add(audit)
            await session.commit()
        except Exception as e:
            logger.warning(f"Failed to save retrieval audit: {e}")

        return RetrieveResponse(
            query=request.query,
            retrieval_mode=retrieval_mode,
            contexts=[ContextItem(**c) for c in contexts],
            debug=debug,
        )

    except Exception as e:
        logger.exception(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
