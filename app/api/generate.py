"""Answer generation API endpoint for LinearRag-RAG."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from app.api.auth import verify_api_key
from app.storage.db import get_db_session
from sqlalchemy.ext.asyncio import AsyncSession
from app.storage.search_repository import SearchRepository
from app.retrieval.retriever import retrieve
from app.retrieval.packer import pack_contexts
from app.embedding.factory import create_embedding_runtime
from app.generator import generate_answer
from app.settings import get_settings
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class GenerateRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, le=100)
    mode: str = Field(default="hybrid_ppr", description="hybrid_ppr, hybrid_bfs, hybrid, vector_only, bm25_only")
    max_tokens: int = Field(default=512, le=2048)


class GenerateResponse(BaseModel):
    query: str
    answer: str
    contexts: List[Dict[str, Any]]
    retrieval_mode: str
    debug: Dict[str, Any]


@router.post("/retrieve_and_answer", response_model=GenerateResponse)
async def retrieve_and_answer(
    request: GenerateRequest,
    _: None = Depends(verify_api_key),
    session: AsyncSession = Depends(get_db_session),
):
    """Retrieve contexts and generate answer using LLM."""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")

    settings = get_settings()

    try:
        # Get query embedding
        query_embedding = None
        if request.mode in ("hybrid_ppr", "hybrid_bfs", "hybrid", "vector_only"):
            try:
                embedder = create_embedding_runtime(settings)
                embeddings = embedder.create_embeddings([request.query])
                if embeddings:
                    query_embedding = embeddings[0]
            except Exception as e:
                logger.warning(f"Embedding failed, falling back to BM25: {e}")

        # Retrieve contexts
        repo = SearchRepository(session)
        if request.mode == "linear_rag":
            from app.retrieval.linear_rag import LinearRagRetriever
            # For LinearRag, embedder is required inside retriever.retrieve but we pass it during init
            embedder = create_embedding_runtime(settings)
            linear_retriever = LinearRagRetriever(repo=repo, embedding_model=embedder)
            result = await linear_retriever.retrieve(
                query=request.query,
                top_k=request.top_k,
            )
        else:
            result = await retrieve(
                repo=repo,
                query=request.query,
                top_k=request.top_k,
                retrieval_mode=request.mode,
                query_embedding=query_embedding,
            )

        packed = pack_contexts(result.contexts, top_k=request.top_k)

        # Generate answer
        answer = generate_answer(
            query=request.query,
            contexts=packed,
            max_tokens=request.max_tokens,
        )

        return GenerateResponse(
            query=request.query,
            answer=answer,
            contexts=packed,
            retrieval_mode=result.retrieval_mode,
            debug=result.debug,
        )

    except Exception as e:
        logger.exception(f"Retrieve and answer failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
