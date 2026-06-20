from fastapi import APIRouter
from app.settings import get_settings

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok"}


@router.get("/metadata")
async def metadata():
    settings = get_settings()
    return {
        "service": "linear_rag-rag",
        "ppr_alpha": settings.PPR_ALPHA,
        "ppr_max_iter": settings.PPR_MAX_ITER,
        "ppr_max_frontier_nodes": settings.PPR_MAX_FRONTIER_NODES,
        "ppr_max_frontier_edges": settings.PPR_MAX_FRONTIER_EDGES,
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_dimensions": settings.EMBEDDING_DIMENSIONS,
        "chunk_size_tokens": settings.CHUNK_SIZE_TOKENS,
        "chunk_overlap_tokens": settings.CHUNK_OVERLAP_TOKENS,
    }
