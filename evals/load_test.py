#!/usr/bin/env python3
"""Concurrency load test for LinearRag-RAG.
Tests concurrent request handling and memory stability.

Usage:
    python evals/load_test.py --concurrent 50 --requests 200
    python evals/load_test.py --concurrent 100 --requests 500 --mode hybrid_ppr
"""
import asyncio
import httpx
import time
import argparse
import statistics
from typing import List, Dict, Any

BASE_URL = "http://localhost:8002"
API_KEY = "test-secret-key"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

QUERIES = [
    "ERR_AUTH_REDIRECT_MISMATCH",
    "tenant_8392 login gagal",
    "AcmeID callback URL",
    "payment-service error",
    "release 2026.06 update",
    "allowlist tenant configuration",
    "SSO provider integration",
    "error code 403 troubleshooting",
]


async def run_single_request(client: httpx.AsyncClient, query: str, mode: str) -> Dict[str, Any]:
    """Run a single retrieval request."""
    start = time.monotonic()
    try:
        r = await client.post(f"{BASE_URL}/retrieve", json={
            "query": query,
            "top_k": 5,
            "mode": mode,
        })
        latency = (time.monotonic() - start) * 1000
        return {
            "status": r.status_code,
            "latency_ms": latency,
            "contexts": len(r.json().get("contexts", [])) if r.status_code == 200 else 0,
            "error": None,
        }
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        return {
            "status": 0,
            "latency_ms": latency,
            "contexts": 0,
            "error": str(e),
        }


async def run_load_test(concurrent: int, total_requests: int, mode: str):
    """Run concurrent load test."""
    print("=" * 60)
    print("LinearRag-RAG Concurrency Load Test")
    print(f"Concurrent: {concurrent}, Total: {total_requests}, Mode: {mode}")
    print("=" * 60)

    # Verify server
    async with httpx.AsyncClient(timeout=30.0, headers=HEADERS) as client:
        try:
            r = await client.get(f"{BASE_URL}/health")
            if r.status_code != 200:
                print("Server not healthy!")
                return
        except Exception:
            print(f"Server not reachable at {BASE_URL}")
            return

    # Run load test
    semaphore = asyncio.Semaphore(concurrent)
    results: List[Dict[str, Any]] = []

    async def bounded_request(client: httpx.AsyncClient, query: str, mode: str):
        async with semaphore:
            return await run_single_request(client, query, mode)

    print(f"\nRunning {total_requests} requests...")
    start_time = time.monotonic()

    async with httpx.AsyncClient(timeout=60.0, headers=HEADERS) as client:
        tasks = []
        for i in range(total_requests):
            query = QUERIES[i % len(QUERIES)]
            tasks.append(bounded_request(client, query, mode))

        results = await asyncio.gather(*tasks)

    total_time = (time.monotonic() - start_time) * 1000

    # Analyze results
    statuses = [r["status"] for r in results]
    latencies = [r["latency_ms"] for r in results]
    errors = [r for r in results if r["error"] or r["status"] != 200]

    success_count = sum(1 for s in statuses if s == 200)
    error_count = len(errors)

    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    avg_latency = statistics.mean(latencies)
    qps = total_requests / (total_time / 1000)

    print("\n" + "=" * 60)
    print("LOAD TEST RESULTS")
    print("=" * 60)
    print(f"Total Requests:  {total_requests}")
    print(f"Successful:      {success_count} ({success_count/total_requests:.1%})")
    print(f"Failed:          {error_count}")
    print(f"Total Time:      {total_time:.0f}ms")
    print(f"QPS:             {qps:.1f}")
    print(f"\nLatency:")
    print(f"  Avg:           {avg_latency:.0f}ms")
    print(f"  P50:           {p50:.0f}ms")
    print(f"  P95:           {p95:.0f}ms")
    print(f"  P99:           {p99:.0f}ms")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:5]:
            print(f"  status={e['status']} latency={e['latency_ms']:.0f}ms error={e['error']}")

    # Acceptance criteria
    print(f"\n--- Acceptance Criteria ---")
    checks = [
        ("Success rate >= 99%", success_count / total_requests >= 0.99),
        ("P50 < 200ms", p50 < 200),
        ("P95 < 500ms", p95 < 500),
        ("P99 < 1000ms", p99 < 1000),
    ]
    for label, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {label}")


def main():
    parser = argparse.ArgumentParser(description="LinearRag-RAG Load Test")
    parser.add_argument("--concurrent", type=int, default=50, help="Max concurrent requests")
    parser.add_argument("--requests", type=int, default=200, help="Total requests to send")
    parser.add_argument("--mode", choices=["bm25_only", "hybrid", "hybrid_ppr"], default="hybrid_ppr")
    parser.add_argument("--url", default=BASE_URL, help="Server URL")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.url

    asyncio.run(run_load_test(args.concurrent, args.requests, args.mode))


if __name__ == "__main__":
    main()
