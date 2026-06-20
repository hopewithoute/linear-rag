import asyncio
import httpx
import json
import time
import os
import sys

API_URL = "http://localhost:8003"
HEADERS = {"Authorization": "Bearer test-secret-key"}

async def eval_retrieval(dataset_name):
    questions_path = f"/var/www/grasp-next/services/linear-rag/dataset/{dataset_name}/questions.json"
    
    if not os.path.exists(questions_path):
        print(f"Error: Could not find {questions_path}")
        return

    print(f"Loading {dataset_name} questions...")
    with open(questions_path, 'r') as f:
        questions = json.load(f)
        
    print(f"Loaded {len(questions)} questions for {dataset_name}. Starting Retrieval Evaluation...")
    
    hits_5 = 0
    total_latency = 0.0
    successful_requests = 0
    
    # Process sequentially or with small concurrency
    sem = asyncio.Semaphore(10) 
    
    async def process_question(i, q, client):
        nonlocal hits_5, total_latency, successful_requests
        gold_ans = str(q['answer']).lower()
        query = q['question']
        
        async with sem:
            start_time = time.time()
            
            try:
                resp = await client.post(
                    f"{API_URL}/retrieve", 
                    json={"query": query, "top_k": 5}, 
                    headers=HEADERS,
                    timeout=30.0
                )
                if resp.status_code != 200:
                    return
            except Exception as e:
                print(f"Retrieval error: {e}")
                return
                
            data = resp.json()
            contexts = data.get("contexts", [])
            
            hit = False
            for rank, ctx in enumerate(contexts):
                text = ctx.get("text", "")
                if gold_ans in text.lower():
                    hit = True
            
            latency = time.time() - start_time
            
            return latency, hit, query

    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [process_question(i, q, client) for i, q in enumerate(questions)]
        
        # gather with progress
        for i, task in enumerate(asyncio.as_completed(tasks)):
            res = await task
            if res:
                lat, h, _ = res
                total_latency += lat
                successful_requests += 1
                if h: hits_5 += 1
                
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(questions)}...")
                
    print(f"\n--- {dataset_name.upper()} RETRIEVAL RESULTS ---")
    print(f"Total Questions Evaluated: {successful_requests}")
    
    if successful_requests > 0:
        hit_rate = hits_5 / successful_requests
        avg_latency = total_latency / successful_requests * 1000
        print(f"Hit@5: {hit_rate:.2%}")
        print(f"Average Latency: {avg_latency:.2f} ms")
        return hit_rate, avg_latency
    return 0, 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_retrieval_only.py <dataset_name>")
        sys.exit(1)
    dataset = sys.argv[1]
    asyncio.run(eval_retrieval(dataset))
