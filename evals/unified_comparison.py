#!/usr/bin/env python3
"""Unified RAGAS Comparison: LinearRag-RAG vs Lazy-Graph-RAG.
Uses same eval script and metrics for fair comparison.

Usage:
    python evals/unified_comparison.py --samples 10
"""
import asyncio
import json
import math
import os
import sys
import time
import httpx
import argparse
import random
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock

# Mock missing modules
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()
sys.modules['langchain_community.chat_models.vertexai'].ChatVertexAI = MagicMock()

# Load .env files
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# Server configs
FRUGAL_RAG_URL = "http://localhost:8002"
LAZY_GRAPH_RAG_URL = "http://localhost:8000"
API_KEY = "test-secret-key"


def sanitize_for_json(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif hasattr(obj, "item") and callable(obj.item):
        val = obj.item()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    return obj


def load_ragbench(topic: str, num_samples: int) -> tuple[List[str], List[Dict]]:
    """Load RAGBench dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets: pip install datasets")
        return [], []

    print(f"Loading RAGBench dataset for topic '{topic}'...")
    try:
        ds = load_dataset("galileo-ai/ragbench", topic, split="test")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return [], []

    total = len(ds)
    print(f"Dataset loaded. Total samples: {total}")

    if num_samples < total:
        random.seed(42)
        indices = random.sample(range(total), num_samples)
        sampled = ds.select(indices)
    else:
        sampled = ds
        num_samples = total

    all_docs = set()
    qa_pairs = []

    for row in sampled:
        q = row.get("question", row.get("query", ""))
        docs = row.get("documents", [])
        gt = row.get("response", row.get("ground_truth", ""))

        if isinstance(docs, str):
            all_docs.add(docs)
        elif isinstance(docs, list):
            for d in docs:
                if isinstance(d, str):
                    all_docs.add(d)
                elif isinstance(d, dict):
                    all_docs.add(d.get("text", d.get("content", "")))

        qa_pairs.append({"question": q, "ground_truth": gt})

    print(f"Extracted {len(all_docs)} unique documents, {len(qa_pairs)} QA pairs")
    return list(all_docs), qa_pairs


async def ingest_documents_linear_rag(api_url: str, documents: List[str], collection_id: str):
    """Ingest documents into LinearRag-RAG."""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(base_url=api_url, headers=headers, timeout=120.0) as client:
        print(f"Ingesting {len(documents)} documents into LinearRag-RAG...")
        for idx, doc_text in enumerate(documents):
            if not doc_text or len(doc_text.strip()) < 10:
                continue
            resp = await client.post("/ingest/document", json={
                "document_id": f"{collection_id}-{idx}",
                "text": doc_text[:50000],
                "metadata": {"source": "ragbench", "collection": collection_id},
            })
            if resp.status_code == 200 and idx % 10 == 0:
                d = resp.json()
                print(f"  [{idx}/{len(documents)}] chunks={d['chunks_created']}")

        print("Compiling graph...")
        resp = await client.post("/graph/compile", json={"mode": "full", "publish_on_success": True})
        if resp.status_code == 200:
            d = resp.json()
            print(f"Compiled: snap={d.get('snapshot_id')}, chunks={d.get('num_chunk_nodes')}")


async def run_linear_rag_queries(api_url: str, qa_pairs: List[Dict], mode: str) -> List[Dict]:
    """Run queries on LinearRag-RAG."""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    results = []

    async with httpx.AsyncClient(base_url=api_url, headers=headers, timeout=60.0) as client:
        for idx, qa in enumerate(qa_pairs):
            q = qa["question"]
            if not q or len(q.strip()) < 3:
                continue

            start = time.monotonic()
            try:
                resp = await client.post("/retrieve_and_answer", json={
                    "query": q,
                    "top_k": 5,
                    "mode": mode,
                })
                latency_ms = (time.monotonic() - start) * 1000

                if resp.status_code != 200:
                    continue

                data = resp.json()
                results.append({
                    "question": q,
                    "answer": data.get("answer", ""),
                    "contexts": [ctx.get("text", "") for ctx in data.get("contexts", [])],
                    "ground_truth": qa.get("ground_truth", ""),
                    "latency_ms": latency_ms,
                })

                if idx % 5 == 0:
                    print(f"  [{idx}/{len(qa_pairs)}] {latency_ms:.0f}ms")

            except Exception as e:
                print(f"  [{idx}] Error: {e}")

    return results


async def run_lgs_queries(api_url: str, qa_pairs: List[Dict]) -> List[Dict]:
    """Run queries on Lazy-Graph-RAG."""
    headers = {"Authorization": f"Bearer local-dev-key", "Content-Type": "application/json"}
    results = []

    async with httpx.AsyncClient(base_url=api_url, headers=headers, timeout=60.0) as client:
        for idx, qa in enumerate(qa_pairs):
            q = qa["question"]
            if not q or len(q.strip()) < 3:
                continue

            start = time.monotonic()
            try:
                resp = await client.post("/v1/query", json={
                    "tenantId": "eval-tenant",
                    "collectionId": "ragbench_covidqa_eval",
                    "query": q,
                    "topK": 5,
                    "budgetPreset": "lite",
                })
                latency_ms = (time.monotonic() - start) * 1000

                if resp.status_code != 200:
                    continue

                data = resp.json()
                results.append({
                    "question": q,
                    "answer": data.get("answer", ""),
                    "contexts": [ctx.get("content", "") for ctx in data.get("contexts", [])],
                    "ground_truth": qa.get("ground_truth", ""),
                    "latency_ms": latency_ms,
                })

                if idx % 5 == 0:
                    print(f"  [{idx}/{len(qa_pairs)}] {latency_ms:.0f}ms")

            except Exception as e:
                print(f"  [{idx}] Error: {e}")

    return results


def run_ragas_evaluation(results: List[Dict], system_name: str) -> Dict[str, float]:
    """Run RAGAS metrics on results."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, context_precision, context_recall
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        print(f"RAGAS dependencies missing: {e}")
        return {}

    # Filter valid entries
    valid = [r for r in results if r.get("ground_truth") and r.get("contexts")]
    if not valid:
        print(f"No valid results for {system_name}")
        return {}

    dataset_dict = {
        "question": [r["question"] for r in valid],
        "answer": [r.get("answer", "") for r in valid],
        "contexts": [r.get("contexts", []) for r in valid],
        "ground_truth": [r["ground_truth"] for r in valid],
    }

    eval_dataset = Dataset.from_dict(dataset_dict)

    gen_api_key = os.environ.get("GENERATOR_API_KEY")
    gen_base_url = os.environ.get("GENERATOR_BASE_URL")
    gen_model = os.environ.get("GENERATOR_MODEL")

    if not gen_api_key or not gen_base_url or not gen_model:
        print("Missing GENERATOR config")
        return {}

    llm = ChatOpenAI(model=gen_model, api_key=gen_api_key, base_url=gen_base_url, max_retries=3)
    metrics = [faithfulness, context_precision, context_recall]

    print(f"Running RAGAS evaluation for {system_name}...")
    try:
        results = evaluate(eval_dataset, metrics=metrics, llm=llm)
        scores = results if isinstance(results, dict) else results.to_pandas().to_dict(orient="records")
        return sanitize_for_json(scores)
    except Exception as e:
        print(f"RAGAS evaluation failed: {e}")
        return {}


async def run_comparison(topic: str, num_samples: int):
    """Run unified comparison."""
    print("=" * 60)
    print("LinearRag-RAG vs Lazy-Graph-RAG Comparison")
    print(f"Topic: {topic}, Samples: {num_samples}")
    print("=" * 60)

    # Load dataset
    documents, qa_pairs = load_ragbench(topic, num_samples)
    if not documents or not qa_pairs:
        print("No data loaded!")
        return

    collection_id = f"ragbench_{topic}"

    # Ingest into LinearRag-RAG
    await ingest_documents_linear_rag(FRUGAL_RAG_URL, documents, collection_id)

    # Run queries on both systems
    print("\n--- Running LinearRag-RAG queries ---")
    linear_rag_results = await run_linear_rag_queries(FRUGAL_RAG_URL, qa_pairs, "hybrid_ppr")

    print("\n--- Running Lazy-Graph-RAG queries ---")
    lgs_results = await run_lgs_queries(LAZY_GRAPH_RAG_URL, qa_pairs)

    # Run RAGAS evaluation
    print("\n--- RAGAS Evaluation ---")
    linear_rag_scores = run_ragas_evaluation(linear_rag_results, "LinearRag-RAG")
    lgs_scores = run_ragas_evaluation(lgs_results, "Lazy-Graph-RAG")

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    metrics = ["faithfulness", "context_precision", "context_recall"]
    print(f"{'Metric':<20} {'LinearRag-RAG':>12} {'Lazy-Graph-RAG':>15}")
    print("-" * 50)

    # Handle both dict and list formats
    linear_rag_flat = linear_rag_scores[0] if isinstance(linear_rag_scores, list) and linear_rag_scores else linear_rag_scores if isinstance(linear_rag_scores, dict) else {}
    lgs_flat = lgs_scores[0] if isinstance(lgs_scores, list) and lgs_scores else lgs_scores if isinstance(lgs_scores, dict) else {}

    for metric in metrics:
        linear_rag_val = linear_rag_flat.get(metric, "N/A")
        lgs_val = lgs_flat.get(metric, "N/A")

        linear_rag_str = f"{linear_rag_val:.3f}" if isinstance(linear_rag_val, (int, float)) else str(linear_rag_val)
        lgs_str = f"{lgs_val:.3f}" if isinstance(lgs_val, (int, float)) else str(lgs_val)

        print(f"{metric:<20} {linear_rag_str:>12} {lgs_str:>15}")

    # Save results
    export = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "topic": topic,
        "samples": num_samples,
        "linear_rag_rag": {
            "scores": linear_rag_scores,
            "avg_latency_ms": sum(r["latency_ms"] for r in linear_rag_results) / len(linear_rag_results) if linear_rag_results else 0,
        },
        "lazy_graph_rag": {
            "scores": lgs_scores,
            "avg_latency_ms": sum(r["latency_ms"] for r in lgs_results) / len(lgs_results) if lgs_results else 0,
        },
    }

    export_path = Path("evals/unified_comparison_results.json")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, "w") as f:
        json.dump(export, f, indent=2, default=str)
    print(f"\nResults saved to {export_path}")


def main():
    global FRUGAL_RAG_URL, LAZY_GRAPH_RAG_URL
    parser = argparse.ArgumentParser(description="Unified RAGAS Comparison")
    parser.add_argument("--topic", default="covidqa", help="RAGBench topic")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples")
    parser.add_argument("--linear_rag-url", default=FRUGAL_RAG_URL, help="LinearRag-RAG URL")
    parser.add_argument("--lgs-url", default=LAZY_GRAPH_RAG_URL, help="Lazy-Graph-RAG URL")
    args = parser.parse_args()

    FRUGAL_RAG_URL = args.linear_rag_url
    LAZY_GRAPH_RAG_URL = args.lgs_url

    asyncio.run(run_comparison(args.topic, args.samples))


if __name__ == "__main__":
    main()
