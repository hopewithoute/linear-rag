import asyncio
import time
import httpx

async def main():
    docs = [
        f"This is a dummy document {i}. It talks about the global pandemic and the effects of COVID-19 on the economy. " * 10
        for i in range(10)
    ]
    
    start = time.time()
    async with httpx.AsyncClient() as client:
        for idx, doc in enumerate(docs):
            await client.post(
                "http://localhost:8002/ingest/document",
                json={
                    "document_id": f"dummy-{idx}",
                    "text": doc,
                    "metadata": {"source": "test"}
                }
            )
        
        # compile
        await client.post("http://localhost:8002/graph/compile", json={"mode": "full", "publish_on_success": True})
        
    elapsed = time.time() - start
    print(f"Ingested {len(docs)} documents in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    asyncio.run(main())
