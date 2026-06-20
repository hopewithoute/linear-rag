import asyncio
import httpx
import json
import time

API_URL = "http://localhost:8003"
HEADERS = {"Authorization": "Bearer test-secret-key"}

async def eval_retrieval():
    print("Loading 2WikiMultihop questions...")
    with open('/var/www/grasp-next/services/linear-rag/dataset/2wikimultihop/questions.json', 'r') as f:
        questions = json.load(f)
        
    print(f"Loaded {len(questions)} questions. Starting evaluation...")
    
    hits_5 = 0
    mrr_5 = 0.0
    total_latency = 0.0
    successful_requests = 0
    
    sem = asyncio.Semaphore(10)
    
    async def fetch_and_eval(i, q, client):
        nonlocal hits_5, mrr_5, total_latency, successful_requests
        gold_ans = str(q['answer']).lower()
        query = q['question']
        
        async with sem:
            start_time = time.time()
            try:
                resp = await client.post(
                    f"{API_URL}/retrieve", 
                    json={"query": query, "top_k": 5}, 
                    headers=HEADERS
                )
                if resp.status_code != 200:
                    return
            except Exception:
                return
                
            latency = time.time() - start_time
            
            data = resp.json()
            contexts = data.get("contexts", [])
            
            hit = False
            local_mrr = 0.0
            for rank, ctx in enumerate(contexts):
                if gold_ans in ctx.get("text", "").lower():
                    hit = True
                    local_mrr = 1.0 / (rank + 1)
                    break
                    
            return latency, hit, local_mrr

    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [fetch_and_eval(i, q, client) for i, q in enumerate(questions)]
        results = await asyncio.gather(*tasks)
        
        for res in results:
            if res:
                lat, h, mrr = res
                total_latency += lat
                successful_requests += 1
                if h:
                    hits_5 += 1
                mrr_5 += mrr
                
    print("\n--- FINAL EVALUATION RESULTS ---")
    print(f"Total Questions Evaluated: {successful_requests}")
    print(f"Hit@5: {hits_5 / successful_requests:.2%}")
    print(f"MRR@5: {mrr_5 / successful_requests:.4f}")
    print(f"Average Latency: {total_latency / successful_requests * 1000:.2f} ms")

if __name__ == "__main__":
    asyncio.run(eval_retrieval())
