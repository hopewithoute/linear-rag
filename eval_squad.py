import asyncio
import json
import time
import requests
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from app.settings import get_settings

def wait_for_api():
    print("Waiting for LinearRAG API...")
    while True:
        try:
            requests.get("http://127.0.0.1:8765/health", timeout=2)
            break
        except:
            time.sleep(2)
    print("API is ready.")

async def ingest_squad():
    print("Loading SQuAD dataset...")
    ds = load_dataset("rajpurkar/squad", split="validation")
    
    unique_contexts = list(dict.fromkeys(ds["context"]))
    print(f"Found {len(unique_contexts)} unique paragraphs (chunks).")
    
    context_to_id = {ctx: str(i) for i, ctx in enumerate(unique_contexts)}
    
    # Ingest in batches
    batch_size = 50
    for i in tqdm(range(0, len(unique_contexts), batch_size), desc="Ingesting SQuAD"):
        batch = unique_contexts[i:i+batch_size]
        for ctx in batch:
            req_data = {
                "tenant_id": "squad_eval",
                "source_uri": f"squad_{context_to_id[ctx]}",
                "text": ctx,
                "metadata": {}
            }
            resp = requests.post(
                "http://127.0.0.1:8765/ingest/document", 
                headers={"Authorization": "Bearer test-secret-key"},
                json=req_data
            )
            resp.raise_for_status()

    # Collect QA pairs
    qa_pairs = []
    for item in ds:
        qa_pairs.append({
            "question": item["question"],
            "target_chunk_id": f"squad_{context_to_id[item['context']]}"
        })
    return qa_pairs

async def eval_squad(qa_pairs, mode="hybrid"):
    print(f"Evaluating 1000 questions using {mode}...")
    hit_5 = 0
    latencies = []
    
    # Evaluate a sample of 1000 questions for speed
    import random
    random.seed(42)
    sample_qa = random.sample(qa_pairs, min(250, len(qa_pairs)))
    
    for qa in tqdm(sample_qa, desc=f"Eval {mode}"):
        t0 = time.time()
        resp = requests.post(
            "http://127.0.0.1:8765/retrieve",
            headers={"Authorization": "Bearer test-secret-key"},
            json={
                "query": qa["question"],
                "tenant_id": "squad_eval",
                "top_k": 5,
                "mode": mode
            }
        )
        t1 = time.time()
        latencies.append((t1 - t0) * 1000)
        
        if resp.status_code == 200:
            hits = resp.json().get("contexts", [])
            retrieved_sources = [doc_to_uri.get(hit.get("document_id")) for hit in hits]
            if qa["target_chunk_id"] in retrieved_sources:
                hit_5 += 1
                
    acc = hit_5 / len(sample_qa) * 100
    avg_lat = np.mean(latencies)
    print(f"\n--- SQuAD {mode.upper()} RESULTS ---")
    print(f"Hit@5: {acc:.2f}%")
    print(f"Average Latency: {avg_lat:.2f} ms")
    return acc

async def main():
    from app.storage.db import init_db
    import app.storage.db as db
    from sqlalchemy import text
    
    init_db()
    
    # Check if data exists
    async with db.engine.connect() as conn:
        result = await conn.execute(text("SELECT count(*) FROM linear_rag.documents WHERE tenant_id='squad_eval'"))
        count = result.scalar()
        
    if count == 0:
        print("Wiping existing squad_eval data...")
        async with db.engine.begin() as conn:
            await conn.execute(text("""
                DELETE FROM linear_rag.mentions WHERE chunk_id IN (
                    SELECT c.chunk_id FROM linear_rag.chunks c 
                    JOIN linear_rag.documents d ON c.document_id = d.id 
                    WHERE d.tenant_id='squad_eval'
                )
            """))
            await conn.execute(text("DELETE FROM linear_rag.chunks WHERE document_id IN (SELECT id FROM linear_rag.documents WHERE tenant_id='squad_eval')"))
            await conn.execute(text("DELETE FROM linear_rag.documents WHERE tenant_id='squad_eval'"))
            
        wait_for_api()
        qa_pairs = await ingest_squad()
    else:
        print(f"Found {count} documents already ingested. Skipping ingestion.")
        from datasets import load_dataset
        ds = load_dataset("rajpurkar/squad", split="validation")
        unique_contexts = list(dict.fromkeys(ds["context"]))
        context_to_id = {ctx: str(i) for i, ctx in enumerate(unique_contexts)}
        qa_pairs = []
        for item in ds:
            qa_pairs.append({
                "question": item["question"],
                "target_chunk_id": f"squad_{context_to_id[item['context']]}"
            })
        wait_for_api()

    # Pre-fetch document mapping
    async with db.engine.connect() as conn:
        result = await conn.execute(text("SELECT id, source_uri FROM linear_rag.documents WHERE tenant_id='squad_eval'"))
        global doc_to_uri
        doc_to_uri = {row[0]: row[1] for row in result.all()}

    
    # Evaluate BM25 only
    await eval_squad(qa_pairs, mode="bm25_only")
    
    # Evaluate Vector only
    await eval_squad(qa_pairs, mode="vector_only")
    
    # Evaluate Hybrid (Vector + BM25 RRF)
    await eval_squad(qa_pairs, mode="hybrid")

if __name__ == "__main__":
    asyncio.run(main())
