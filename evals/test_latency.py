import asyncio
import json
import httpx
import time

async def main():
    query = "What is the capital of France?"
    
    start = time.time()
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "http://localhost:8002/retrieve",
            json={
                "query": query,
                "top_k": 5,
                "mode": "linear_rag",
                "tenant_id": "test_tenant"
            }
        )
    
    elapsed = time.time() - start
    print(f"Total time: {elapsed:.3f}s")
    
    try:
        data = response.json()
        debug = data.get("debug", {})
        print(json.dumps(debug, indent=2))
    except Exception as e:
        print("Response:", response.text)

if __name__ == "__main__":
    asyncio.run(main())
