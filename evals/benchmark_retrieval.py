import asyncio
import json
import httpx
import time
import numpy as np

async def main():
    # Mix of simple and multi-hop queries related to general knowledge & medical
    queries = [
        "What are the symptoms of COVID-19?",
        "When did the global pandemic start?",
        "What is the relationship between SARS-CoV-2 and bats?",
        "Who is the author of the Harry Potter series?",
        "Tell me about the history of the United States.",
        "What are the long term effects of viral infections?",
        "How do vaccines work to prevent disease?",
        "Who directed the movie Inception?",
        "What is the mechanism of action for mRNA vaccines?",
        "Are there treatments for severe respiratory distress?"
    ]
    
    latencies = []
    print("Benchmarking pure retrieval (no LLM generation)...\n")
    
    async with httpx.AsyncClient(timeout=30.0, headers={"Authorization": "Bearer test-secret-key"}) as client:
        # Warmup
        await client.post(
            "http://localhost:8002/retrieve",
            json={
                "query": "warmup",
                "top_k": 5,
                "mode": "linear_rag",
            }
        )
        
        for q in queries:
            start = time.perf_counter()
            resp = await client.post(
                "http://localhost:8002/retrieve",
                json={
                    "query": q,
                    "top_k": 5,
                    "mode": "linear_rag",
                }
            )
            elapsed = (time.perf_counter() - start) * 1000
            
            if resp.status_code == 200:
                data = resp.json()
                contexts = data.get("contexts", [])
                debug = data.get("debug", {})
                print(f"Query: '{q}'")
                print(f"  -> {len(contexts)} contexts retrieved in {elapsed:.1f}ms")
                # Add to latencies
                latencies.append(elapsed)
            else:
                print(f"Query: '{q}' -> Failed with HTTP {resp.status_code}")
                
    if latencies:
        avg_latency = np.mean(latencies)
        p90 = np.percentile(latencies, 90)
        print("\n--- RESULTS ---")
        print(f"Total queries: {len(latencies)}")
        print(f"Average Latency: {avg_latency:.1f} ms")
        print(f"P90 Latency:     {p90:.1f} ms")
        print(f"Min Latency:     {min(latencies):.1f} ms")
        print(f"Max Latency:     {max(latencies):.1f} ms")

if __name__ == "__main__":
    asyncio.run(main())
