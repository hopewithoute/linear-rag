from fastapi import FastAPI
from app.api import health, ingest, retrieve, graph_api, documents, generate
from app.graph.snapshot import load_snapshot, set_active_snapshot
from app.settings import get_settings
from app.middleware import RequestLoggingMiddleware
import logging
import sys

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("linear_rag-rag")

app = FastAPI(title="LinearRag-RAG Engine", description="Multi-hop context retrieval via sparse graph walking")

# Middleware
app.add_middleware(RequestLoggingMiddleware)

# Routers
app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(retrieve.router)
app.include_router(graph_api.router)
app.include_router(documents.router)
app.include_router(generate.router)


@app.on_event("startup")
async def load_active_snapshot():
    settings = get_settings()
    try:
        from app.storage.db import init_db, get_db_session
        from sqlalchemy import select
        from app.storage.models import GraphSnapshot

        init_db()
        async for session in get_db_session():
            stmt = select(GraphSnapshot).where(GraphSnapshot.active == True).order_by(GraphSnapshot.snapshot_id.desc()).limit(1)
            result = await session.execute(stmt)
            entry = result.scalar_one_or_none()
            if entry:
                gs = load_snapshot(entry.snapshot_id, settings.SNAPSHOT_DIR)
                if gs:
                    set_active_snapshot(gs)
                    logger.info(f"Loaded active snapshot: id={gs.snapshot_id}, chunks={gs.chunk_count}, entities={gs.entity_count}")
                else:
                    logger.warning(f"Snapshot file not found for snapshot_id={entry.snapshot_id}")
            else:
                logger.info("No active snapshot found in database")
    except Exception as e:
        logger.warning(f"Failed to load active snapshot on startup: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
