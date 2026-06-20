import asyncio
from sqlalchemy import text
from app.storage.db import init_db
import app.storage.db as db

async def main():
    init_db()
    async with db.engine.connect() as conn:
        result = await conn.execute(text("SELECT id, source_uri FROM linear_rag.documents WHERE tenant_id='squad_eval' LIMIT 5"))
        doc_to_uri = {row[0]: row[1] for row in result.all()}
        print(f"doc_to_uri: {doc_to_uri}")
        
    import requests
    resp = requests.post(
        "http://127.0.0.1:8765/retrieve",
        headers={"Authorization": "Bearer test-secret-key"},
        json={
            "query": "What is the capital of France?",
            "tenant_id": "squad_eval",
            "top_k": 5
        }
    )
    print(resp.status_code)
    hits = resp.json().get("contexts", [])
    print(f"hits: {hits}")
    retrieved_sources = [doc_to_uri.get(hit.get("document_id")) for hit in hits]
    print(f"retrieved_sources: {retrieved_sources}")

if __name__ == "__main__":
    asyncio.run(main())
