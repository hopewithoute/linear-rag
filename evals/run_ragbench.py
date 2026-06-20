#!/usr/bin/env python3
"""LinearRag-RAG RAGBench Evaluation Harness.
Uses galileo-ai/ragbench dataset for standardized RAG evaluation.

Usage:
    python evals/run_ragbench.py --topic covidqa --samples 10
    python evals/run_ragbench.py --topic techqa --samples 20 --skip-ingest
"""
import asyncio
import json
import logging
import re
import string
import sys
import time
import httpx
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8002"
API_KEY = "test-secret-key"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


async def _retry_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    **kwargs,
) -> httpx.Response:
    """Make an HTTP request with exponential backoff retry.

    Retries up to 3 times on TimeoutException, ConnectError, or HTTPStatusError
    for 5xx responses. Does NOT retry 4xx errors.
    """
    delays = [1.0, 2.0, 4.0]
    last_exc: Exception | None = None

    for attempt, delay in enumerate(delays):
        try:
            resp = await client.request(method, url, **kwargs)
            if resp.status_code < 500:
                # 4xx and 2xx — do not retry
                return resp
            # 5xx — retry
            last_exc = httpx.HTTPStatusError(
                f"Server error {resp.status_code}: {resp.text}",
                request=resp.request,
                response=resp,
            )
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            last_exc = exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code < 500:
                # 4xx — do not retry
                raise
            last_exc = exc
        except Exception as exc:
            last_exc = exc

        if attempt < len(delays) - 1:
            logger.warning("Request failed (attempt %d/%d), retrying in %.1fs: %s",
                           attempt + 1, len(delays), delay, last_exc)
            await asyncio.sleep(delay)
        else:
            logger.error("Request failed after %d attempts: %s", len(delays), last_exc)

    raise last_exc if last_exc else RuntimeError("Unexpected retry failure")


async def load_ragbench(topic: str, num_samples: int) -> tuple[List[str], List[Dict]]:
    """Load RAGBench dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        sys.exit(1)

    logger.info(f"Loading RAGBench dataset for topic '{topic}'...")
    try:
        ds = await asyncio.to_thread(load_dataset, "galileo-ai/ragbench", topic, split="test")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    total = len(ds)
    logger.info(f"Dataset loaded. Total samples: {total}")

    if num_samples < total:
        random.seed(42)
        indices = random.sample(range(total), num_samples)
        sampled = ds.select(indices)
    else:
        sampled = ds
        num_samples = total

    # Extract documents and questions
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

    logger.info(f"Extracted {len(all_docs)} unique documents, {len(qa_pairs)} QA pairs")
    return list(all_docs), qa_pairs


async def ingest_documents(api_url: str, documents: List[str], collection_id: str):
    """Ingest documents into LinearRag-RAG."""
    headers = HEADERS.copy()
    ingest_failures = 0

    async with httpx.AsyncClient(base_url=api_url, headers=headers, timeout=120.0) as client:
        # Check if already ingested
        try:
            r = await _retry_request(client, "GET", "/graph/snapshot/active")
            if r.status_code == 200 and r.json():
                snap = r.json()
                if snap.get("num_chunk_nodes", 0) > 0:
                    logger.info(f"Snapshot already exists (id={snap['snapshot_id']}, chunks={snap['num_chunk_nodes']}). Skipping ingest.")
                    return
        except Exception:
            pass

        logger.info(f"Ingesting {len(documents)} documents...")
        for idx, doc_text in enumerate(documents):
            if not doc_text or len(doc_text.strip()) < 10:
                continue

            resp = await _retry_request(client, "POST", "/ingest/document", json={
                "document_id": f"ragbench-{collection_id}-{idx}",
                "text": doc_text[:50000],  # Truncate very long docs
                "metadata": {"source": "ragbench", "collection": collection_id, "index": idx},
            })

            if resp.status_code != 200:
                logger.warning(f"  Failed to ingest doc {idx}: {resp.text[:100]}")
                ingest_failures += 1
            elif idx % 10 == 0:
                d = resp.json()
                logger.info(f"  [{idx}/{len(documents)}] chunks={d['chunks_created']}, entities={d['entities_extracted']}")

        # Compile graph
        logger.info("Compiling graph...")
        resp = await _retry_request(client, "POST", "/graph/compile", json={"mode": "full", "publish_on_success": True})
        if resp.status_code != 200:
            logger.error(f"Compile failed: {resp.status_code} {resp.text}")
            raise RuntimeError(f"Graph compile failed: {resp.status_code} {resp.text}")

        d = resp.json()
        logger.info(f"Compiled: snap={d.get('snapshot_id')}, chunks={d.get('num_chunk_nodes')}, entities={d.get('num_entity_nodes')}, edges={d.get('num_directed_edges')}")

    # Summary
    success_count = len(documents) - ingest_failures
    logger.info(f"Ingest summary: {success_count}/{len(documents)} documents ingested successfully")
    if ingest_failures > 0 and ingest_failures == len(documents):
        logger.error("All document ingestions failed")
        raise RuntimeError("All document ingestions failed")


async def run_queries(api_url: str, qa_pairs: List[Dict], mode: str) -> List[Dict]:
    """Run queries and collect results."""
    headers = HEADERS.copy()
    results = []

    async with httpx.AsyncClient(base_url=api_url, headers=headers, timeout=120.0) as client:
        for idx, qa in enumerate(qa_pairs):
            q = qa["question"]
            if not q or len(q.strip()) < 3:
                continue

            start = time.monotonic()
            try:
                resp = await _retry_request(client, "POST", "/retrieve", json={
                    "query": q,
                    "top_k": 5,
                    "mode": mode,
                })
                latency_ms = (time.monotonic() - start) * 1000

                if resp.status_code != 200:
                    logger.warning(f"  [{idx}] Error: HTTP {resp.status_code}")
                    continue

                data = resp.json()
                contexts = data.get("contexts", [])
                debug = data.get("debug", {})

                results.append({
                    "question": q,
                    "answer": "",  # LinearRag-RAG doesn't generate answers yet
                    "contexts": [ctx.get("text", "") for ctx in contexts],
                    "ground_truth": qa.get("ground_truth", ""),
                    "latency_ms": latency_ms,
                    "debug": debug,
                })

                if idx % 5 == 0:
                    logger.info(f"  [{idx}/{len(qa_pairs)}] {len(contexts)} contexts, {latency_ms:.0f}ms")

            except Exception as e:
                logger.error(f"  [{idx}] Exception: {e}")

    return results


def compute_metrics(results: List[Dict], k: int = 5) -> Dict[str, float]:
    """Compute retrieval metrics."""
    if not results:
        return {}

    total = len(results)
    hit_count = 0
    mrr_sum = 0.0
    recall_sum = 0.0

    for r in results:
        contexts = r.get("contexts", [])
        gt = r.get("ground_truth", "").lower()

        if not gt:
            continue

        # Hit@K: does any context contain part of the ground truth?
        # Strip punctuation for cleaner word matching
        gt_clean = re.sub(rf"[{re.escape(string.punctuation)}]", "", gt)
        gt_words = set(gt_clean.split())
        hit = False
        rr = 0.0
        recall = 0.0

        for idx, ctx in enumerate(contexts[:k]):
            ctx_lower = ctx.lower()
            # Strip punctuation from context too for fair matching
            ctx_clean = re.sub(rf"[{re.escape(string.punctuation)}]", "", ctx_lower)
            # Check if ground truth words appear in context
            found_words = sum(1 for w in gt_words if w in ctx_clean)
            word_overlap = found_words / len(gt_words) if gt_words else 0

            if word_overlap > 0.3:  # 30% word overlap threshold
                hit = True
                if rr == 0:
                    rr = 1.0 / (idx + 1)
                recall = max(recall, word_overlap)

        if hit:
            hit_count += 1
        mrr_sum += rr
        recall_sum += recall

    return {
        "total_queries": total,
        f"hit@{k}": hit_count / total if total else 0,
        "mrr": mrr_sum / total if total else 0,
        "recall": recall_sum / total if total else 0,
        "avg_latency_ms": sum(r["latency_ms"] for r in results) / total if total else 0,
    }


async def run_ragbench_eval(topic: str, num_samples: int, mode: str, skip_ingest: bool):
    """Run full RAGBench evaluation."""
    logger.info("=" * 60)
    logger.info(f"LinearRag-RAG RAGBench Evaluation")
    logger.info(f"Topic: {topic}, Samples: {num_samples}, Mode: {mode}")
    logger.info("=" * 60)

    collection_id = f"ragbench_{topic}"

    # Load dataset
    documents, qa_pairs = await load_ragbench(topic, num_samples)
    if not documents or not qa_pairs:
        logger.error("No data loaded!")
        sys.exit(1)

    # Ingest documents
    if not skip_ingest:
        await ingest_documents(BASE_URL, documents, collection_id)
    else:
        logger.info("Skipping ingestion (--skip-ingest)")

    # Run queries
    logger.info(f"\nRunning {len(qa_pairs)} queries with mode '{mode}'...")
    results = await run_queries(BASE_URL, qa_pairs, mode)

    if not results:
        logger.error("No results!")
        sys.exit(1)

    # Compute metrics
    metrics = compute_metrics(results, k=5)

    logger.info("\n" + "=" * 60)
    logger.info("RAGBENCH EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Topic:       {topic}")
    logger.info(f"Mode:        {mode}")
    logger.info(f"Queries:     {metrics.get('total_queries', 0)}")
    logger.info(f"Hit@5:       {metrics.get('hit@5', 0):.1%}")
    logger.info(f"MRR:         {metrics.get('mrr', 0):.3f}")
    logger.info(f"Recall:      {metrics.get('recall', 0):.3f}")
    logger.info(f"Avg Latency: {metrics.get('avg_latency_ms', 0):.0f}ms")

    # Save results - export ALL results, not just first 20
    export = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "topic": topic,
        "mode": mode,
        "metrics": metrics,
        "details": results,
    }

    export_path = Path(f"evals/ragbench_{topic}_{mode}_results.json")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, "w") as f:
        json.dump(export, f, indent=2, default=str)
    logger.info(f"\nResults saved to {export_path}")


def main():
    global BASE_URL
    parser = argparse.ArgumentParser(description="LinearRag-RAG RAGBench Eval")
    parser.add_argument("--topic", default="covidqa",
                        help="RAGBench topic (covidqa, finqa, techqa, etc.)")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of samples to evaluate")
    parser.add_argument("--mode", choices=["bm25_only", "hybrid", "linear_rag", "vector_only"], default="linear_rag",
                        help="Retrieval mode")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip document ingestion (use existing data)")
    parser.add_argument("--url", default=BASE_URL,
                        help="LinearRag-RAG server URL")
    args = parser.parse_args()

    BASE_URL = args.url

    asyncio.run(run_ragbench_eval(args.topic, args.samples, args.mode, args.skip_ingest))


if __name__ == "__main__":
    main()