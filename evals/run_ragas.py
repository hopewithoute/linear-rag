#!/usr/bin/env python3
"""LinearRag-RAG RAGAS Evaluation Harness.
Supports multiple collections (covidqa, HotpotQA, etc.)

Usage:
    python evals/run_ragas.py --topic covidqa --samples 10 --mode hybrid_ppr
    python evals/run_ragas.py --compare --topic covidqa --samples 10
    python evals/run_ragas.py --ragas-only evals/ragbench_covidqa_hybrid_ppr_results.json
"""
import argparse
import asyncio
import json
import logging
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock

import httpx

warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_ragas")

# Safely mock langchain_community.chat_models.vertexai only if it actually fails to import
# due to missing optional dependencies (google-cloud-aiplatform, etc.)
try:
    from langchain_community.chat_models.vertexai import ChatVertexAI  # noqa: F401
except ImportError as e:
    if "vertexai" in str(e) or "google" in str(e):
        logger.debug("Mocking unavailable langchain_community.chat_models.vertexai: %s", e)
        sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()
        sys.modules["langchain_community.chat_models.vertexai"].ChatVertexAI = MagicMock()
    else:
        raise

from dataclasses import dataclass

BASE_URL = "http://localhost:8002"
API_KEY = "test-secret-key"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


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
        logger.error("Please install datasets: pip install datasets")
        sys.exit(1)

    logger.info("Loading RAGBench dataset for topic '%s'...", topic)
    try:
        ds = load_dataset("galileo-ai/ragbench", topic, split="test")
    except Exception as e:
        logger.error("Failed to load dataset: %s", e)
        sys.exit(1)

    total = len(ds)
    logger.info("Dataset loaded. Total samples: %d", total)

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
            stripped = docs.strip()
            if stripped:
                all_docs.add(stripped)
        elif isinstance(docs, list):
            for d in docs:
                if isinstance(d, str):
                    stripped = d.strip()
                    if stripped:
                        all_docs.add(stripped)
                elif isinstance(d, dict):
                    text = d.get("text", d.get("content", "")).strip()
                    if text:
                        all_docs.add(text)

        qa_pairs.append({"question": q, "ground_truth": gt})

    logger.info("Extracted %d unique documents, %d QA pairs", len(all_docs), len(qa_pairs))
    return list(all_docs), qa_pairs


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


async def ingest_documents(api_url: str, documents: List[str], collection_id: str, force: bool = False):
    """Ingest documents into LinearRag-RAG."""
    headers = HEADERS.copy()

    async with httpx.AsyncClient(base_url=api_url, headers=headers, timeout=120.0) as client:
        if not force:
            try:
                r = await _retry_request(client, "GET", "/graph/snapshot/active")
                if r.status_code == 200 and r.json():
                    snap = r.json()
                    if snap.get("num_chunk_nodes", 0) > 0:
                        logger.info("Snapshot exists (id=%s, chunks=%s). Skipping ingest.",
                                    snap["snapshot_id"], snap["num_chunk_nodes"])
                        return
            except Exception:
                pass

        logger.info("Ingesting %d documents for collection '%s'...", len(documents), collection_id)
        fail_count = 0
        for idx, doc_text in enumerate(documents):
            if not doc_text or len(doc_text.strip()) < 10:
                continue

            text_to_ingest = doc_text
            if len(doc_text) > 50000:
                logger.warning("Truncating document %d from %d to 50000 chars", idx, len(doc_text))
                text_to_ingest = doc_text[:50000]

            resp = await _retry_request(client, "POST", "/ingest/document", json={
                "document_id": f"{collection_id}-doc-{idx}",
                "text": text_to_ingest,
                "metadata": {"source": "ragbench", "collection": collection_id},
            })

            if resp.status_code != 200:
                fail_count += 1
                logger.warning("Ingest failed for doc %d: %s %s", idx, resp.status_code, resp.text[:200])
            elif idx % 10 == 0:
                d = resp.json()
                logger.info("  [%d/%d] chunks=%s", idx, len(documents), d.get("chunks_created", "?"))

        total_ingested = len(documents) - fail_count
        logger.info("Ingest complete: %d succeeded, %d failed out of %d", total_ingested, fail_count, len(documents))
        if fail_count == len(documents):
            logger.error("All documents failed to ingest!")
            sys.exit(1)

        logger.info("Compiling graph...")
        resp = await _retry_request(client, "POST", "/graph/compile", json={"mode": "full", "publish_on_success": True})
        if resp.status_code != 200:
            logger.error("Graph compile failed: %s %s", resp.status_code, resp.text[:500])
            sys.exit(1)
        d = resp.json()
        logger.info("Compiled: snap=%s, chunks=%s", d.get("snapshot_id"), d.get("num_chunk_nodes"))


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
            latency_ms = 0.0
            try:
                resp = await _retry_request(client, "POST", "/retrieve", json={
                    "query": q,
                    "top_k": 5,
                    "mode": mode,
                })
                latency_ms = (time.monotonic() - start) * 1000

                if resp.status_code != 200:
                    logger.warning("Query %d failed: HTTP %s body=%s", idx, resp.status_code, resp.text[:500])
                    results.append({
                        "question": q,
                        "answer": "",
                        "contexts": [],
                        "ground_truth": qa.get("ground_truth", ""),
                        "latency_ms": latency_ms,
                        "error": f"HTTP {resp.status_code}",
                    })
                    continue

                data = resp.json()
                contexts = data.get("contexts", [])

                results.append({
                    "question": q,
                    "answer": "",  # LinearRag-RAG doesn't generate answers
                    "contexts": [ctx.get("text", "") for ctx in contexts],
                    "ground_truth": qa.get("ground_truth", ""),
                    "latency_ms": latency_ms,
                })

                if idx % 5 == 0:
                    logger.info("  [%d/%d] %d contexts, %.0fms", idx, len(qa_pairs), len(contexts), latency_ms)

            except httpx.TimeoutException as e:
                latency_ms = (time.monotonic() - start) * 1000
                logger.warning("Query %d timed out: %s", idx, e)
                results.append({"question": q, "answer": "", "contexts": [], "ground_truth": qa.get("ground_truth", ""), "latency_ms": latency_ms, "error": "timeout"})
            except httpx.ConnectError as e:
                latency_ms = (time.monotonic() - start) * 1000
                logger.warning("Query %d connection error: %s", idx, e)
                results.append({"question": q, "answer": "", "contexts": [], "ground_truth": qa.get("ground_truth", ""), "latency_ms": latency_ms, "error": "connect_error"})
            except httpx.HTTPError as e:
                latency_ms = (time.monotonic() - start) * 1000
                logger.warning("Query %d HTTP error: %s", idx, e)
                results.append({"question": q, "answer": "", "contexts": [], "ground_truth": qa.get("ground_truth", ""), "latency_ms": latency_ms, "error": str(e)[:200]})

    return results


def run_ragas_evaluation(results_path: Path):
    """Run RAGAS metrics on exported results."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, context_precision, context_recall
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError as e:
        print(f"RAGAS dependencies missing: {e}")
        print("Install with: pip install ragas langchain-openai")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    details = data.get("details", [])
    if not details:
        print("No query results found!")
        return

    # Filter out entries without ground_truth
    valid = [r for r in details if r.get("ground_truth") and len(r["ground_truth"].strip()) > 0]
    if not valid:
        print("No entries with ground_truth!")
        return

    questions = [r["question"] for r in valid]
    answers = [r.get("answer", "") for r in valid]
    contexts = [r.get("contexts", []) for r in valid]
    ground_truths = [r["ground_truth"] for r in valid]

    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }

    eval_dataset = Dataset.from_dict(dataset_dict)

    # Generator LLM config
    gen_api_key = os.environ.get("GENERATOR_API_KEY")
    gen_base_url = os.environ.get("GENERATOR_BASE_URL")
    gen_model = os.environ.get("GENERATOR_MODEL")

    if not gen_api_key or not gen_base_url or not gen_model:
        print("Missing GENERATOR config. Set GENERATOR_API_KEY, GENERATOR_BASE_URL, GENERATOR_MODEL")
        return

    print(f"Initializing Generator LLM ({gen_model})...")
    llm = ChatOpenAI(
        model=gen_model,
        api_key=gen_api_key,
        base_url=gen_base_url,
        max_retries=3,
    )

    metrics = [faithfulness, context_precision, context_recall]

    # Optional: AnswerRelevancy with embeddings
    enable_relevancy = os.environ.get("RAGAS_ENABLE_ANSWER_RELEVANCY", "").lower() == "true"
    embeddings = None
    if enable_relevancy:
        emb_api_key = os.environ.get("EMBEDDING_API_KEY")
        emb_base_url = os.environ.get("EMBEDDING_BASE_URL")
        emb_model = os.environ.get("EMBEDDING_MODEL")
        if emb_api_key and emb_base_url and emb_model:
            embeddings = OpenAIEmbeddings(
                model=emb_model,
                api_key=emb_api_key,
                base_url=emb_base_url,
            )
            from ragas.metrics import AnswerRelevancy
            try:
                metrics.append(AnswerRelevancy(llm=llm, embeddings=embeddings))
            except TypeError:
                # RAGAS version that does not accept llm= param
                metrics.append(AnswerRelevancy(embeddings=embeddings))

    logger.info("Running RAGAS evaluation on %d queries...", len(valid))
    try:
        results = evaluate(eval_dataset, metrics=metrics, llm=llm, embeddings=embeddings)

        print("\n" + "=" * 60)
        print("RAGAS EVALUATION RESULTS")
        print("=" * 60)
        print(results)

        export_path = results_path.parent / f"ragas_{results_path.stem}.json"

        # Robust fallback chain for results serialization
        results_dict = None
        if isinstance(results, dict):
            results_dict = results
        else:
            try:
                results_dict = results.to_pandas().to_dict(orient="records")
            except Exception:
                try:
                    results_dict = dict(results)
                except Exception:
                    if hasattr(results, "__dict__"):
                        results_dict = {k: v for k, v in results.__dict__.items() if not k.startswith("_")}
                    else:
                        results_dict = {"result": str(results)}

        clean_results = sanitize_for_json(results_dict)

        with open(export_path, "w") as f:
            json.dump(clean_results, f, indent=2, allow_nan=False)
        logger.info("RAGAS scores saved to %s", export_path)

    except Exception as e:
        logger.error("RAGAS evaluation failed: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)


async def run_full_eval(topic: str, num_samples: int, mode: str, skip_ingest: bool, force_ingest: bool = False):
    """Run full evaluation: ingest → query → RAGAS."""
    print("=" * 60)
    print("LinearRag-RAG RAGAS Evaluation")
    print(f"Topic: {topic}, Samples: {num_samples}, Mode: {mode}")
    print("=" * 60)

    collection_id = f"ragbench_{topic}"

    # Load dataset
    documents, qa_pairs = load_ragbench(topic, num_samples)
    if not documents or not qa_pairs:
        logger.error("No data loaded for topic='%s'!", topic)
        sys.exit(1)

    # Ingest
    if not skip_ingest:
        await ingest_documents(BASE_URL, documents, collection_id, force=force_ingest)

    # Run queries
    logger.info("Running %d queries...", len(qa_pairs))
    results = await run_queries(BASE_URL, qa_pairs, mode)

    if not results:
        logger.error("No results returned!")
        sys.exit(1)

    # Compute basic metrics
    total = len(results)
    hit_count = sum(1 for r in results if any(
        w in " ".join(r["contexts"]).lower()
        for w in r.get("ground_truth", "").lower().split()[:5]
    ))
    avg_latency = sum(r["latency_ms"] for r in results) / total

    logger.info("--- Basic Metrics ---")
    logger.info("Queries:     %d", total)
    logger.info("Hit@5:       %d/%d (%.1f%%)", hit_count, total, hit_count/total*100)
    logger.info("Avg Latency: %.0fms", avg_latency)

    # Export for RAGAS
    export = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "topic": topic,
        "mode": mode,
        "collection_id": collection_id,
        "metrics": {
            "total_queries": total,
            "hit_at_k": hit_count / total,
            "avg_latency_ms": avg_latency,
        },
        "details": results,
    }

    export_path = Path(f"evals/ragbench_{topic}_{mode}_results.json")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, "w") as f:
        json.dump(export, f, indent=2)
    logger.info("Results exported to %s", export_path)

    # Run RAGAS evaluation
    logger.info("--- Running RAGAS Evaluation ---")
    run_ragas_evaluation(export_path)


async def run_comparison_eval(topic: str, num_samples: int, skip_ingest: bool, force_ingest: bool = False):
    """Run 4-way RAGAS comparison: bm25_only vs vector_only vs hybrid vs hybrid_ppr."""
    modes = ["bm25_only", "vector_only", "hybrid", "hybrid_ppr"]
    print("=" * 60)
    print("LinearRag-RAG 3-Way RAGAS Comparison")
    print(f"Topic: {topic}, Samples: {num_samples}, Modes: {modes}")
    print("=" * 60)

    collection_id = f"ragbench_{topic}"

    documents, qa_pairs = load_ragbench(topic, num_samples)
    if not documents or not qa_pairs:
        logger.error("No data loaded for topic='%s'!", topic)
        sys.exit(1)

    if not skip_ingest:
        await ingest_documents(BASE_URL, documents, collection_id, force=force_ingest)

    all_scores = {}
    for mode in modes:
        print(f"\n--- Running mode: {mode} ---")
        results = await run_queries(BASE_URL, qa_pairs, mode)
        if not results:
            logger.warning("  No results for %s", mode)
            continue

        export_path = Path(f"evals/ragbench_{topic}_{mode}_results.json")
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "topic": topic,
            "mode": mode,
            "collection_id": collection_id,
            "details": results,
        }
        with open(export_path, "w") as f:
            json.dump(export, f, indent=2)

        run_ragas_evaluation(export_path)
        ragas_path = export_path.parent / f"ragas_{export_path.stem}.json"
        if ragas_path.exists():
            with open(ragas_path) as f:
                all_scores[mode] = json.load(f)

    # Comparison table
    if len(all_scores) >= 2:
        print(f"\n{'=' * 70}")
        print("RAGAS COMPARISON TABLE")
        print(f"{'=' * 70}")
        print(f"{'Mode':<15} {'Faithfulness':>12} {'Ctx Precision':>14} {'Ctx Recall':>12}")
        print("-" * 55)

        baseline = None
        for mode, scores in all_scores.items():
            flat = scores[0] if isinstance(scores, list) else scores
            faith = flat.get("faithfulness", "N/A")
            prec = flat.get("context_precision", "N/A")
            rec = flat.get("context_recall", "N/A")

            if baseline is None:
                baseline = {"prec": prec, "rec": rec}

            delta = ""
            if baseline and mode != list(all_scores.keys())[0] and isinstance(prec, (int, float)):
                dp = prec - baseline["prec"]
                dr = rec - baseline["rec"]
                delta = f"  ΔP{dp:+.3f} ΔR{dr:+.3f}"

            faith_str = f"{faith:.3f}" if isinstance(faith, (int, float)) else str(faith)
            prec_str = f"{prec:.3f}" if isinstance(prec, (int, float)) else str(prec)
            rec_str = f"{rec:.3f}" if isinstance(rec, (int, float)) else str(rec)

            print(f"{mode:<15} {faith_str:>12} {prec_str:>14} {rec_str:>12}{delta}")


def main():
    global BASE_URL
    parser = argparse.ArgumentParser(description="LinearRag-RAG RAGAS Eval")
    parser.add_argument("--topic", default="covidqa", help="RAGBench topic (covidqa, finqa, techqa)")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--mode", choices=["bm25_only", "hybrid", "hybrid_ppr", "vector_only"], default="hybrid_ppr")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion")
    parser.add_argument("--force-ingest", action="store_true", help="Force re-ingestion even if snapshot exists")
    parser.add_argument("--ragas-only", type=str, default=None, help="Run RAGAS only on existing results file")
    parser.add_argument("--compare", action="store_true", help="Run 3-way comparison")
    parser.add_argument("--url", default=BASE_URL, help="Server URL")
    args = parser.parse_args()

    BASE_URL = args.url

    if args.ragas_only:
        run_ragas_evaluation(Path(args.ragas_only))
    elif args.compare:
        asyncio.run(run_comparison_eval(args.topic, args.samples, args.skip_ingest, args.force_ingest))
    else:
        asyncio.run(run_full_eval(args.topic, args.samples, args.mode, args.skip_ingest, args.force_ingest))


if __name__ == "__main__":
    main()
