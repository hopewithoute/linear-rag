#!/usr/bin/env python3
"""LinearRag-RAG Eval aligned with lazy-graph-rag pattern.
Uses same golden set format and metrics (Hit@K, MRR).
"""
import asyncio
import json
import logging
import sys
import httpx
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

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

    # Exhausted retries
    raise last_exc if last_exc else RuntimeError("Unexpected retry failure")


# --- Ingest -------------------------------------------------------------------

async def ingest_fixtures(api_url: str, golden_set_path: Path, *, skip_ingest: bool = False):
    """Ingest fixture documents.

    Args:
        api_url: Base URL of the server.
        golden_set_path: Path to the golden set JSON file.
        skip_ingest: If True, skip ingestion entirely (useful when an active
                     snapshot is already present and up-to-date).
    """
    try:
        with open(golden_set_path, "r") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        logger.error("Golden set not found: %s", golden_set_path)
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in golden set %s: %s", golden_set_path, e)
        sys.exit(1)

    fixtures_dir = golden_set_path.parent
    sources_indexed: set[str] = set()

    async with httpx.AsyncClient(base_url=api_url, headers=HEADERS, timeout=120.0) as client:
        # Check for existing active snapshot to avoid redundant ingestion
        if skip_ingest:
            logger.info("skip_ingest flag set, skipping snapshot check and ingest.")
        else:
            try:
                snap_resp = await _retry_request(client, "GET", "/graph/snapshot/active")
                if snap_resp.status_code == 200 and snap_resp.json():
                    snap = snap_resp.json()
                    if snap.get("num_chunk_nodes", 0) > 0:
                        logger.info(
                            "Active snapshot exists with %d chunks, skipping ingest.",
                            snap.get("num_chunk_nodes"),
                        )
                        return
            except Exception as e:
                logger.warning("Could not check active snapshot, proceeding with ingest: %s", e)

        for item in dataset:
            source_file = item.get("sourceFixture")
            if not source_file or source_file in sources_indexed:
                continue

            file_path = fixtures_dir / source_file
            if file_path.exists():
                logger.info("Indexing fixture: %s", source_file)
                try:
                    with open(file_path, "r") as f:
                        content = f.read()
                except OSError as e:
                    logger.error("Failed to read fixture %s: %s", source_file, e)
                    continue

                try:
                    resp = await _retry_request(client, "POST", "/ingest/document", json={
                        "document_id": f"fixture-{source_file}",
                        "text": content,
                        "metadata": {"type": "fixture", "source": source_file},
                    })
                    if resp.status_code == 200:
                        sources_indexed.add(source_file)
                    else:
                        logger.warning("Failed to index %s: %s", source_file, resp.text)
                except Exception as e:
                    logger.error("Error indexing %s: %s", source_file, e)

        # Compile graph
        logger.info("Compiling graph...")
        try:
            resp = await _retry_request(client, "POST", "/graph/compile", json={
                "mode": "full",
                "publish_on_success": True,
            })
            if resp.status_code == 200:
                d = resp.json()
                logger.info(
                    "Compiled: %s chunks, %s entities",
                    d.get("num_chunk_nodes", "?"),
                    d.get("num_entity_nodes", "?"),
                )
            else:
                logger.error("Graph compile failed (%d): %s", resp.status_code, resp.text)
                sys.exit(1)
        except Exception as e:
            logger.error("Graph compile request failed: %s", e)
            sys.exit(1)


# --- Eval ---------------------------------------------------------------------

async def run_eval(golden_set_path: Path, mode: str = "hybrid"):
    """Run evaluation aligned with lazy-graph-rag pattern."""
    logger.info("--- Starting Eval (mode=%s) ---", mode)

    try:
        with open(golden_set_path, "r") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        logger.error("Golden set not found: %s", golden_set_path)
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in golden set %s: %s", golden_set_path, e)
        sys.exit(1)

    if not dataset:
        logger.error("Golden set is empty: %s", golden_set_path)
        sys.exit(1)

    results: list[dict[str, Any]] = []
    mrr_sum = 0.0
    hit_at_k_sum = 0
    total = len(dataset)
    k = 5
    failures = 0

    async with httpx.AsyncClient(headers=HEADERS, timeout=60.0) as client:
        for item in dataset:
            question = item.get("question")
            if not question:
                logger.warning("Skipping item with missing 'question': %s", item)
                continue

            expected_hints = item.get("expectedChunkHints", [])
            reference_answer = item.get("referenceAnswer", "")

            logger.info("Query: %s", question)

            try:
                resp = await _retry_request(client, "POST", f"{BASE_URL}/retrieve", json={
                    "query": question,
                    "top_k": k,
                    "mode": mode,
                })
            except Exception as e:
                logger.error("Request failed for question '%s': %s", question, e)
                failures += 1
                continue

            if resp.status_code != 200:
                logger.warning("Non-200 response for '%s': %s %s",
                               question, resp.status_code, resp.text)
                failures += 1
                continue

            data = resp.json()
            contexts = data.get("contexts", [])

            # Metric: Hit@K and MRR based on expected hints
            hit = False
            reciprocal_rank = 0.0

            for idx, ctx in enumerate(contexts):
                content = ctx.get("text", "").lower()
                found = any(hint.lower() in content for hint in expected_hints)
                if found:
                    hit = True
                    reciprocal_rank = 1.0 / (idx + 1)
                    break

            if hit:
                hit_at_k_sum += 1
            mrr_sum += reciprocal_rank

            status = "HIT" if hit else "MISS"
            logger.info("  %s hit=%s mrr=%.2f contexts=%d", status, hit, reciprocal_rank, len(contexts))

            # Export format for RAGAS
            results.append({
                "question": question,
                "answer": "",  # LinearRag-RAG doesn't generate answers
                "contexts": [ctx.get("text", "") for ctx in contexts],
                "ground_truth": reference_answer,
            })

    logger.info("\n--- Eval Results ---")
    logger.info("Total queries: %d", total)
    if failures:
        logger.warning("Failed queries: %d/%d", failures, total)

    if total == 0:
        logger.error("No valid queries to evaluate.")
        sys.exit(1)

    hit_rate = hit_at_k_sum / total
    mrr_rate = mrr_sum / total
    logger.info("Hit@%d: %.2f", k, hit_rate)
    logger.info("MRR: %.2f", mrr_rate)

    export_path = Path(f"evals/aligned_eval_{mode}_results.json")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(export_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Exported results to %s", export_path.absolute())
    except OSError as e:
        logger.error("Failed to write results to %s: %s", export_path, e)
        sys.exit(1)

    return {
        "mode": mode,
        "total": total,
        "hit_at_k": hit_rate,
        "mrr": mrr_rate,
        "failures": failures,
    }


# --- Comparison ---------------------------------------------------------------

async def run_comparison(golden_set_path: Path):
    """Run comparison across all modes.

    WARNING: Running multiple modes sequentially may be affected by server-side
    state mutations (e.g., graph changes between modes). For fully independent
    results, restart the server between modes.
    """
    logger.warning(
        "WARNING: Running multiple modes sequentially may be affected by server-side "
        "state mutations. For fully independent results, restart the server between modes."
    )
    logger.info("=" * 60)
    logger.info("LinearRag-RAG Aligned Eval (lazy-graph-rag pattern)")
    logger.info("=" * 60)

    modes = ["bm25_only", "vector_only", "hybrid", "hybrid_bfs", "hybrid_ppr"]
    all_results: dict[str, dict[str, Any]] = {}

    for mode in modes:
        logger.info("--- Mode: %s ---", mode)
        result = await run_eval(golden_set_path, mode)
        all_results[mode] = result

    # Comparison table
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON TABLE")
    logger.info("=" * 60)
    logger.info("%-15s %8s %8s %8s", "Mode", "Hit@5", "MRR", "Failed")
    logger.info("-" * 45)

    for mode, result in all_results.items():
        logger.info(
            "%-15s %7.1f%% %8.3f %8d",
            mode,
            result["hit_at_k"] * 100,
            result["mrr"],
            result.get("failures", 0),
        )


# --- CLI ----------------------------------------------------------------------

def main():
    import argparse
    global BASE_URL

    parser = argparse.ArgumentParser(description="LinearRag-RAG Aligned Eval")
    parser.add_argument(
        "--golden-set",
        type=str,
        default=None,
        help="Path to golden set JSON",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "bm25_only", "vector_only", "hybrid",
            "hybrid_bfs", "hybrid_ppr", "all",
        ],
        default="all",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest fixtures before eval",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip ingestion step when --ingest is set (use existing snapshot)",
    )
    parser.add_argument(
        "--url",
        default=BASE_URL,
        help="Server URL",
    )
    args = parser.parse_args()

    BASE_URL = args.url

    golden_path = (
        Path(args.golden_set)
        if args.golden_set
        else Path("tests/fixtures/eval_golden_set.json")
    )

    if args.ingest:
        asyncio.run(ingest_fixtures(BASE_URL, golden_path, skip_ingest=args.skip_ingest))

    if args.mode == "all":
        asyncio.run(run_comparison(golden_path))
    else:
        asyncio.run(run_eval(golden_path, args.mode))


if __name__ == "__main__":
    main()