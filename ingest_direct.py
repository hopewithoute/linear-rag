import asyncio
import json
import time
from sqlalchemy import text
from app.storage import db
from app.api.ingest import ingest_document, IngestRequest
from app.settings import get_settings
from app.extraction.factory import create_term_extractor
from pydantic import BaseModel
from app.api.ingest import ingest_document, IngestRequest

async def main():
    print("Loading dataset...")
    chunks_path = "/var/www/grasp-next/services/linear-rag/dataset/2wikimultihop/chunks_10k.json"
    
    with open(chunks_path, "r") as f:
        passages = json.load(f)
        
    print(f"Loaded {len(passages)} passages from dataset.")
    
    settings = get_settings()
    
    db.init_db()
    
    start_time = time.time()
    
    print("Clearing database...")
    async with db.AsyncSessionLocal() as session:
        await session.execute(text(f"TRUNCATE TABLE {settings.DB_SCHEMA}.documents CASCADE"))
        await session.commit()
    print("Database cleared.")

    batch_size = 500
    for i in range(0, len(passages), batch_size):
        batch = passages[i:i+batch_size]
        print(f"Ingesting batch {i//batch_size + 1}/{(len(passages)//batch_size + (1 if len(passages)%batch_size!=0 else 0))} ({len(batch)} passages)...")
        
        async with db.AsyncSessionLocal() as session:
            for j, passage_text in enumerate(batch):
                req = IngestRequest(
                    text=passage_text,
                    source_uri=f"2wiki_{i+j}",
                    metadata={},
                    tenant_id=""
                )
                await ingest_document(req, None, session)
            await session.commit()
            
    print(f"All ingested in {time.time() - start_time:.2f}s")
    
    # Compile graph
    from app.api.graph_api import compile_snapshot, CompileRequest
    print("Compiling graph...")
    async with db.AsyncSessionLocal() as session:
        req = CompileRequest(mode="full", publish=True)
        resp = await compile_snapshot(req, None, session)
        print("Graph compiled:", resp)

if __name__ == "__main__":
    asyncio.run(main())
