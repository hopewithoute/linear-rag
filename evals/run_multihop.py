#!/usr/bin/env python3
"""LinearRag-RAG Multi-Hop Evaluation Harness.
Uses HotpotQA dataset for multi-hop reasoning evaluation.

Usage:
    python evals/run_multihop.py --samples 20 --mode hybrid_ppr
    python evals/run_multihop.py --compare --samples 10
"""
import asyncio
import json
import logging
import sys
import time
import httpx
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8002"
API_KEY = "test-secret-key"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


# --- Retry helper -------------------------------------------------------------

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
                return resp
            last_exc = httpx.HTTPStatusError(
                f"Server error {resp.status_code}: {resp.text}",
                request=resp.request,
                response=resp,
            )
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            last_exc = exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code < 500:
                raise
            last_exc = exc
        except Exception as exc:
            last_exc = exc

        if attempt < len(delays) - 1:
            logger.warning(
                "Request failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1, len(delays), delay, last_exc,
            )
            await asyncio.sleep(delay)
        else:
            logger.error("Request failed after %d attempts: %s", len(delays), last_exc)

    raise last_exc if last_exc else RuntimeError("Unexpected retry failure")


# --- Data classes -------------------------------------------------------------

@dataclass
class MultiHopResult:
    question_id: str
    question: str
    mode: str
    hop_type: str
    gold_answer: str
    predicted_answer: str
    contexts: List[Dict[str, Any]]
    hit_at_k: bool
    answer_match: bool
    latency_ms: float


# --- Dataset loading ----------------------------------------------------------

def load_hotpotqa(num_samples: int = 20, split: str = "validation") -> tuple[List[str], List[Dict]]:
    """Load HotpotQA dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        sys.exit(1)

    logger.info("Loading HotpotQA dataset (split=%s)...", split)
    try:
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
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

    all_docs: set[str] = set()
    qa_pairs: list[Dict] = []

    for row in sampled:
        q = row.get("question", "")
        answer = row.get("answer", "")
        type_ = row.get("type", "bridge")
        level = row.get("level", "easy")

        supporting_facts = row.get("supporting_facts", [])
        context = row.get("context", {})

        titles = context.get("title", [])
        sentences = context.get("sentences", [])

        doc_texts: list[str] = []
        for title, sents in zip(titles, sentences):
            doc_text = f"Title: {title}\n" + "\n".join(sents)
            doc_texts.append(doc_text)
            all_docs.add(doc_text)

        # Get gold paragraphs
        gold_paragraphs: list[str] = []
        for sf in supporting_facts:
            if isinstance(sf, (list, tuple)) and len(sf) >= 2:
                sf_title, sf_sent_idx = sf[0], sf[1]
                for title, sents in zip(titles, sentences):
                    if title == sf_title and sf_sent_idx < len(sents):
                        gold_paragraphs.append(f"Title: {title}\n{sents[sf_sent_idx]}")

        qa_pairs.append({
            "question": q,
            "answer": answer,
            "type": type_,
            "level": level,
            "gold_paragraphs": gold_paragraphs,
            "gold_keywords": [w.lower() for w in answer.split()],
        })

    logger.info(
        "Extracted %d unique documents, %d QA pairs",
        len(all_docs),
        len(qa_pairs),
    )
    return list(all_docs), qa_pairs


# --- Ingest -------------------------------------------------------------------

async def ingest_documents(
    api_url: str,
    documents: List[str],
    collection_id: str,
    force: bool = False,
):
    """Ingest documents into LinearRag-RAG."""
    headers = HEADERS.copy()

    async with httpx.AsyncClient(base_url=api_url, headers=headers, timeout=120.0) as client:
        if not force:
            try:
                r = await _retry_request(client, "GET", "/graph/snapshot/active")
                if r.status_code == 200 and r.json():
                    snap = r.json()
                    if snap.get("num_chunk_nodes", 0) > 0:
                        logger.info(
                            "Snapshot exists (id=%s, chunks=%s). Skipping ingest.",
                            snap.get("snapshot_id"),
                            snap.get("num_chunk_nodes"),
                        )
                        return
            except Exception:
                pass

        logger.info("Ingesting %d documents...", len(documents))
        ingest_failures = 0
        for idx, doc_text in enumerate(documents):
            d = doc_text
            if not d:
                continue
            d = d.strip()
            if not d or len(d) < 10:
                continue

            try:
                resp = await _retry_request(
                    client, "POST", "/ingest/document",
                    json={
                        "document_id": f"multihop-{collection_id}-{idx}",
                        "text": d[:50000],
                        "metadata": {"source": "hotpotqa", "collection": collection_id},
                    },
                )
                if resp.status_code == 200 and idx % 10 == 0:
                    data = resp.json()
                    logger.info("[%d/%d] chunks=%s", idx, len(documents), data.get("chunks_created"))
                elif resp.status_code != 200:
                    ingest_failures += 1
            except Exception as e:
                logger.warning("[%d] Ingest error: %s", idx, e)
                ingest_failures += 1

        if ingest_failures == len(documents):
            logger.error("All %d documents failed to ingest", ingest_failures)
            sys.exit(1)

        logger.info("Compiling graph...")
        try:
            resp = await _retry_request(client, "POST", "/graph/compile", json={
                "mode": "full",
                "publish_on_success": True,
            })
            if resp.status_code == 200:
                d = resp.json()
                logger.info(
                    "Compiled: snap=%s, chunks=%s",
                    d.get("snapshot_id"),
                    d.get("num_chunk_nodes"),
                )
            else:
                logger.error("Graph compile failed (%d): %s", resp.status_code, resp.text)
                sys.exit(1)
        except Exception as e:
            logger.error("Graph compile request failed: %s", e)
            sys.exit(1)


# --- Query run ----------------------------------------------------------------

async def run_queries(
    api_url: str,
    qa_pairs: List[Dict],
    mode: str,
) -> List[MultiHopResult]:
    """Run queries and collect results."""
    headers = HEADERS.copy()
    results: list[MultiHopResult] = []
    failures = 0

    async with httpx.AsyncClient(base_url=api_url, headers=headers, timeout=120.0) as client:
        for idx, qa in enumerate(qa_pairs):
            q = qa.get("question", "")
            if not q or len(q.strip()) < 3:
                continue

            start = time.monotonic()
            try:
                resp = await _retry_request(
                    client, "POST", "/retrieve_and_answer",
                    json={
                        "query": q,
                        "top_k": 10,
                        "mode": mode,
                    },
                )
                latency_ms = (time.monotonic() - start) * 1000

                if resp.status_code != 200:
                    failures += 1
                    continue

                data = resp.json()
                contexts = data.get("contexts", [])
                predicted_answer = data.get("answer", "")

                # Check if gold keywords appear in contexts
                all_text = " ".join(ctx.get("text", "") for ctx in contexts).lower()
                gold_keywords = qa.get("gold_keywords", [])
                hit = any(kw in all_text for kw in gold_keywords)

                # Check answer match (simple keyword match)
                answer = qa.get("answer", "").lower()
                answer_match = answer in all_text if answer else False

                results.append(MultiHopResult(
                    question_id=f"q{idx}",
                    question=q,
                    mode=mode,
                    hop_type=qa.get("type", "bridge"),
                    gold_answer=qa.get("answer", ""),
                    predicted_answer=predicted_answer,
                    contexts=contexts,
                    hit_at_k=hit,
                    answer_match=answer_match,
                    latency_ms=latency_ms,
                ))

                if idx % 5 == 0:
                    logger.info(
                        "[%d/%d] %d contexts, %.0fms",
                        idx, len(qa_pairs), len(contexts), latency_ms,
                    )

            except Exception as e:
                logger.warning("[%d] Query error: %s", idx, e)
                failures += 1

    total_queries = len(qa_pairs)
    if failures == total_queries:
        logger.error("All %d queries failed", failures)
        sys.exit(1)

    logger.info(
        "Completed %d queries (%d failures)",
        len(results),
        failures,
    )
    return results


# --- Metrics ------------------------------------------------------------------

def compute_metrics(results: List[MultiHopResult]) -> Dict[str, Any]:
    """Compute multi-hop metrics."""
    if not results:
        return {}

    total = len(results)
    hits = sum(1 for r in results if r.hit_at_k)
    answer_matches = sum(1 for r in results if r.answer_match)

    answer_lens = [len(r.predicted_answer) for r in results if r.predicted_answer]
    avg_answer_len = sum(answer_lens) / len(answer_lens) if answer_lens else 0.0
    non_empty_count = len(answer_lens)

    avg_latency = sum(r.latency_ms for r in results) / total

    # Per hop type
    hop_types = set(r.hop_type for r in results)
    per_hop: Dict[str, Dict[str, Any]] = {}
    for ht in hop_types:
        ht_results = [r for r in results if r.hop_type == ht]
        per_hop[ht] = {
            "count": len(ht_results),
            "hit_rate": sum(1 for r in ht_results if r.hit_at_k) / len(ht_results) if ht_results else 0,
            "answer_rate": sum(1 for r in ht_results if r.answer_match) / len(ht_results) if ht_results else 0,
        }

    return {
        "total_queries": total,
        "hit_rate": hits / total,
        "answer_match_rate": answer_matches / total,
        "avg_latency_ms": avg_latency,
        "avg_answer_len": avg_answer_len,
        "non_empty_answer_count": non_empty_count,
        "per_hop_type": per_hop,
    }


# --- Export -------------------------------------------------------------------

def run_ragas_on_results(results: List[MultiHopResult]) -> Dict[str, Any]:
    """Compute RAGAS scores for results with non-empty predicted_answer."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, context_precision, context_recall
    except ImportError:
        logger.warning("ragas not installed, skipping RAGAS scoring")
        return {}

    filtered = [r for r in results if r.predicted_answer and r.predicted_answer.strip()]
    if not filtered:
        logger.info("No results with non-empty predicted_answer for RAGAS scoring")
        return {}

    try:
        from datasets import Dataset
        
        eval_dataset = Dataset.from_list([
            {
                "question": r.question,
                "answer": r.predicted_answer,
                "contexts": [ctx.get("text", "") for ctx in r.contexts],
                "ground_truth": r.gold_answer,
            }
            for r in filtered
        ])

        import os
        api_key = os.environ.get("GENERATOR_API_KEY", "")
        base_url = os.environ.get("GENERATOR_BASE_URL", "https://api.openai.com/v1")
        model = os.environ.get("GENERATOR_MODEL", "gpt-4o-mini")

        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI

        gen_llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)
        gen_wrapper = LangchainLLMWrapper(gen_llm)

        score = evaluate(
            eval_dataset,
            metrics=[faithfulness, context_precision, context_recall],
            llm=gen_wrapper,
        )

        try:
            scores_dict = score.to_pandas().to_dict()
        except Exception:
            try:
                scores_dict = dict(score)
            except Exception:
                scores_dict = {
                    k: getattr(score, k, None)
                    for k in ["faithfulness", "context_precision", "context_recall"]
                }
                
        result = scores_dict

        logger.info("RAGAS scores computed for %d results", len(filtered))
        return result

    except Exception as e:
        logger.warning("RAGAS evaluation failed: %s", e)
        return {}


# --- Main eval ----------------------------------------------------------------

async def run_multihop_eval(
    mode: str = "all",
    num_samples: int = 20,
    skip_ingest: bool = False,
    force_ingest: bool = False,
):
    """Run multi-hop evaluation."""
    logger.info("=" * 60)
    logger.info("LinearRag-RAG Multi-Hop Evaluation")
    logger.info("Samples: %d, Mode: %s", num_samples, mode)
    logger.info("=" * 60)

    modes = ["bm25_only", "vector_only", "hybrid", "hybrid_bfs", "linear_rag"] if mode == "all" else [mode]

    # Load dataset
    documents, qa_pairs = load_hotpotqa(num_samples)
    if not documents or not qa_pairs:
        logger.error("No data loaded!")
        sys.exit(1)

    # Ingest
    if not skip_ingest:
        await ingest_documents(BASE_URL, documents, "hotpotqa", force=force_ingest)

    # Run queries for each mode
    all_results: Dict[str, List[MultiHopResult]] = {}
    all_metrics: Dict[str, Dict[str, Any]] = {}

    for m in modes:
        logger.info("\n--- Running mode: %s ---", m)
        results = await run_queries(BASE_URL, qa_pairs, m)
        all_results[m] = results

        metrics = compute_metrics(results)
        all_metrics[m] = metrics

        ragas = run_ragas_on_results(results)

        logger.info("\n%s Metrics:", m)
        logger.info("  Hit Rate:            %.1f%%", metrics["hit_rate"] * 100)
        logger.info("  Answer Match:        %.1f%%", metrics["answer_match_rate"] * 100)
        logger.info("  Avg Latency:         %.0fms", metrics["avg_latency_ms"])
        logger.info("  Avg Answer Length:   %.1f chars", metrics["avg_answer_len"])
        logger.info("  Non-Empty Answers:   %d/%d", metrics["non_empty_answer_count"], metrics["total_queries"])
        if ragas:
            logger.info("  RAGAS faithfulness:  %s", ragas.get("faithfulness", "N/A"))
            logger.info("  RAGAS ctx_precision:%s", ragas.get("context_precision", "N/A"))
            logger.info("  RAGAS ctx_recall:    %s", ragas.get("context_recall", "N/A"))

        for ht, ht_metrics in metrics.get("per_hop_type", {}).items():
            logger.info(
                "  %s: hit=%.1f%%, answer=%.1f%%",
                ht,
                ht_metrics["hit_rate"] * 100,
                ht_metrics["answer_rate"] * 100,
            )

        all_metrics[m]["ragas_scores"] = ragas

    # Comparison table
    if len(all_metrics) >= 2:
        logger.info("\n" + "=" * 60)
        logger.info("MULTI-HOP COMPARISON")
        logger.info("=" * 60)
        logger.info("%-15s %10s %14s %10s %14s", "Mode", "Hit Rate", "Answer Match", "Latency", "Avg Ans Len")
        logger.info("-" * 65)

        for m, metrics in all_metrics.items():
            logger.info(
                "%-15s %9.1f%% %13.1f%% %9.0fms %13.1f",
                m,
                metrics["hit_rate"] * 100,
                metrics["answer_match_rate"] * 100,
                metrics["avg_latency_ms"],
                metrics["avg_answer_len"],
            )

        # Per-hop comparison
        all_hop_types: set[str] = set()
        for metrics in all_metrics.values():
            all_hop_types.update(metrics.get("per_hop_type", {}).keys())

        if all_hop_types:
            logger.info("\n--- Per Hop Type Comparison ---")
            header = "%-15s" % "Hop Type"
            for m in all_metrics:
                header += f" {m:>12}"
            logger.info(header)
            logger.info("-" * (15 + 13 * len(all_metrics)))

            for ht in sorted(all_hop_types):
                row = f"{ht:<15}"
                for m, metrics in all_metrics.items():
                    ht_metrics = metrics.get("per_hop_type", {}).get(ht, {})
                    hit_rate = ht_metrics.get("hit_rate", 0)
                    row += f" {hit_rate:>11.1%}"
                logger.info(row)

    # Save results
    export: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": "hotpotqa",
        "samples": num_samples,
        "modes": {},
    }
    for m, results in all_results.items():
        export["modes"][m] = {
            "metrics": all_metrics[m],
            "ragas_scores": run_ragas_on_results(results),
            "details": [
                {
                    "id": r.question_id,
                    "question": r.question,
                    "hop_type": r.hop_type,
                    "gold_answer": r.gold_answer,
                    "predicted_answer": r.predicted_answer,
                    "hit": r.hit_at_k,
                    "answer_match": r.answer_match,
                    "latency_ms": round(r.latency_ms),
                    "contexts": len(r.contexts),
                }
                for r in results
            ],
        }

    export_path = Path("evals/multihop_hotpotqa_results.json")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(export_path, "w") as f:
            json.dump(export, f, indent=2, default=str)
        logger.info("\nResults saved to %s", export_path.absolute())
    except OSError as e:
        logger.error("Failed to write results to %s: %s", export_path, e)
        sys.exit(1)


def main():
    global BASE_URL
    parser = argparse.ArgumentParser(description="LinearRag-RAG Multi-Hop Eval")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples")
    parser.add_argument(
        "--mode",
        choices=["bm25_only", "vector_only", "hybrid", "hybrid_bfs", "linear_rag", "all"],
        default="all",
    )
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion")
    parser.add_argument("--force-ingest", action="store_true", help="Force re-ingestion")
    parser.add_argument("--url", default=BASE_URL, help="Server URL")
    args = parser.parse_args()

    BASE_URL = args.url

    asyncio.run(run_multihop_eval(
        mode=args.mode,
        num_samples=args.samples,
        skip_ingest=args.skip_ingest,
        force_ingest=args.force_ingest,
    ))


if __name__ == "__main__":
    main()