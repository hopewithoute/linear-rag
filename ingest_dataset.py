import asyncio
import httpx
import json
import time
import os
import sys
from dotenv import load_dotenv

load_dotenv(".env.mpnet")

API_URL = "http://localhost:8003"
HEADERS = {"Authorization": "Bearer test-secret-key"}

async def main(dataset_name):
    print(f"Loading {dataset_name} dataset...")
    chunks_path = f"/var/www/grasp-next/services/linear-rag/dataset/{dataset_name}/chunks.json"
    
    with open(chunks_path, "r") as f:
        chunks = json.load(f)
        
    print(f"Loaded {len(chunks)} total chunks from dataset.")
    
    start_time = time.time()
    batch_size = 100
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        print(f"Starting ingestion of {len(chunks)} chunks in batches of {batch_size}...")
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            payload_docs = []
            for j, chunk in enumerate(batch):
                text = chunk["text"] if isinstance(chunk, dict) else chunk
                payload_docs.append(text)
            
            combined_text = "\n\n".join(payload_docs)
            
            batch_start = time.time()
            resp = await client.post(
                f"{API_URL}/ingest/document",
                json={"text": combined_text, "source_uri": f"{dataset_name}_batch_{i//batch_size}"},
                headers=HEADERS,
                timeout=300.0
            )
            
            if resp.status_code == 200:
                print(f"Batch {i//batch_size + 1} SUCCESS. Time: {time.time()-batch_start:.2f}s")
            else:
                print(f"Batch {i//batch_size + 1} FAILED: {resp.text}")
                
        print("\nAll batches ingested! Compiling final graph...")
        resp = await client.post(
            f"{API_URL}/graph/compile",
            headers=HEADERS,
            json={},
            timeout=120.0
        )
        if resp.status_code == 200:
            print("Graph compiled successfully!", resp.json())
        else:
            print(f"Graph compile failed: {resp.text}")

    print(f"\nTotal script execution time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_dataset.py <dataset_name>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
