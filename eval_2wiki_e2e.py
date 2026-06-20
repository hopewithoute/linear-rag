import asyncio
import httpx
import json
import time
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load env variables
load_dotenv(".env.mpnet")

API_URL = "http://localhost:8003"
HEADERS = {"Authorization": "Bearer test-secret-key"}

GENERATOR_BASE_URL = os.environ.get("GENERATOR_BASE_URL", "https://api.tokenrouter.com/v1")
GENERATOR_API_KEY = os.environ.get("GENERATOR_API_KEY", "sk-V1vqt1n8b1VTOHGyRjArKfs2WuREdeV5bpvdvJpGaUqNgi8s")
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "MiniMax-M3")

client_llm = AsyncOpenAI(
    api_key=GENERATOR_API_KEY,
    base_url=GENERATOR_BASE_URL
)

PROMPT_TEMPLATE = """You are a helpful assistant for question answering. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Keep the answer as concise as possible, preferably a short phrase or entity name.

Context:
{context}

Question: {question}
Answer:"""

async def eval_e2e():
    print("Loading 2WikiMultihop questions...")
    with open('/var/www/grasp-next/services/linear-rag/dataset/2wikimultihop/questions.json', 'r') as f:
        questions = json.load(f)
        
    print(f"Loaded {len(questions)} questions. Starting End-to-End Evaluation...")
    
    hits_5 = 0
    contain_acc = 0
    total_latency = 0.0
    successful_requests = 0
    
    # Process sequentially or with small concurrency to avoid rate limits
    sem = asyncio.Semaphore(5) 
    
    async def process_question(i, q, client):
        nonlocal hits_5, contain_acc, total_latency, successful_requests
        gold_ans = str(q['answer']).lower()
        query = q['question']
        
        async with sem:
            start_time = time.time()
            
            # 1. Retrieval
            try:
                resp = await client.post(
                    f"{API_URL}/retrieve", 
                    json={"query": query, "top_k": 5}, 
                    headers=HEADERS
                )
                if resp.status_code != 200:
                    return
            except Exception as e:
                print(f"Retrieval error: {e}")
                return
                
            data = resp.json()
            contexts = data.get("contexts", [])
            
            hit = False
            context_text = ""
            for rank, ctx in enumerate(contexts):
                text = ctx.get("text", "")
                # Truncate each document to avoid exceeding 2000 token limit for MiniMax-M3
                trunc_text = text[:800]
                context_text += f"[Document {rank+1}] {trunc_text}\n"
                if gold_ans in text.lower():
                    hit = True
            
            # 2. Generation
            try:
                prompt = PROMPT_TEMPLATE.format(context=context_text, question=query)
                completion = await client_llm.chat.completions.create(
                    model=GENERATOR_MODEL,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=50
                )
                answer = completion.choices[0].message.content.strip().lower()
                
                is_contain = gold_ans in answer
                
            except Exception as e:
                print(f"LLM error: {e}")
                return
                
            latency = time.time() - start_time
            
            return latency, hit, is_contain, answer, gold_ans

    # Let's run a smaller subset first to make it fast and see results, 
    # e.g., first 100 questions to extrapolate, or we can run all 1000 if fast enough.
    # We will run all 1000 since TokenRouter can handle it.
    subset_questions = questions[:]
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [process_question(i, q, client) for i, q in enumerate(subset_questions)]
        
        # gather with progress
        results = []
        for i, task in enumerate(asyncio.as_completed(tasks)):
            res = await task
            if res:
                lat, h, c, ans, gold = res
                total_latency += lat
                successful_requests += 1
                if h: hits_5 += 1
                if c: contain_acc += 1
                
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(subset_questions)}...")
                
    print("\n--- END-TO-END EVALUATION RESULTS ---")
    print(f"Total Questions Evaluated: {successful_requests}")
    print(f"Hit@5 (Retrieval Phase): {hits_5 / successful_requests:.2%}")
    print(f"Contain-Acc. (Generation Phase): {contain_acc / successful_requests:.2%}")
    print(f"Average Total Latency (Ret + Gen): {total_latency / successful_requests * 1000:.2f} ms")

if __name__ == "__main__":
    asyncio.run(eval_e2e())
