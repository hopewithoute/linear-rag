#!/usr/bin/env python3
"""LinearRag-RAG Evaluation Harness.
3-way comparison: BM25-only vs Hybrid (RRF) vs Hybrid PPR (RRF+PPR).

Usage:
    python evals/run_eval.py --mode all
    python evals/run_eval.py --mode all --golden-set evals/multihop_golden.json
    python evals/run_eval.py --mode hybrid --samples 5
"""
import asyncio
import json
import time
import httpx
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

BASE_URL = "http://localhost:8002"
API_KEY = "test-secret-key"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


# ─── Golden Dataset ──────────────────────────────────────────────────────────

GOLDEN_SET = [
    {
        "id": "MH-001", "category": "single_hop", "difficulty": "easy",
        "query": "Apa itu AcmeID?",
        "gold_keywords": ["AcmeID", "SSO", "provider"],
        "requires_ppr": False,
    },
    {
        "id": "MH-002", "category": "exact_keyword", "difficulty": "easy",
        "query": "ERR_AUTH_REDIRECT_MISMATCH",
        "gold_keywords": ["ERR_AUTH_REDIRECT_MISMATCH"],
        "requires_ppr": False,
    },
    {
        "id": "MH-003", "category": "entity_bridge", "difficulty": "medium",
        "query": "Kenapa ERR_AUTH_REDIRECT_MISMATCH terjadi?",
        "gold_keywords": ["callback URL", "AcmeID", "allowlist", "tenant"],
        "requires_ppr": True,
    },
    {
        "id": "MH-004", "category": "two_hop", "difficulty": "medium",
        "query": "Apa root cause login gagal untuk tenant_8392?",
        "gold_keywords": ["callback URL", "release 2026.06", "whitelist", "AcmeID"],
        "requires_ppr": True,
    },
    {
        "id": "MH-005", "category": "entity_bridge", "difficulty": "medium",
        "query": "Service apa saja yang terpengaruh update callback URL?",
        "gold_keywords": ["payment-service", "auth-service", "gateway-service", "port"],
        "requires_ppr": False,
    },
    {
        "id": "MH-006", "category": "temporal_linking", "difficulty": "medium",
        "query": "Kapan error ERR_AUTH_REDIRECT_MISMATCH terjadi dan siapa yang terdampak?",
        "gold_keywords": ["2026-06-15", "tenant_8392", "user_4455"],
        "requires_ppr": False,
    },
    {
        "id": "MH-007", "category": "cross_document_reasoning", "difficulty": "hard",
        "query": "Bagaimana cara fix ERR_AUTH_REDIRECT_MISMATCH untuk tenant_8392?",
        "gold_keywords": ["callback", "/callback/v2", "allowlist", "update"],
        "requires_ppr": True,
    },
    {
        "id": "MH-008", "category": "multi_hop_chain", "difficulty": "hard",
        "query": "Tenant_8392 pakai SSO provider apa dan kenapa login gagal setelah update?",
        "gold_keywords": ["AcmeID", "callback URL", "release 2026.06", "ERR_AUTH_REDIRECT_MISMATCH"],
        "requires_ppr": True,
    },
    {
        "id": "MH-009", "category": "noisy_entity_filter", "difficulty": "hard",
        "query": "Error apa yang terjadi pada payment-service setelah deploy?",
        "gold_keywords": ["ERR_AUTH_REDIRECT_MISMATCH", "payment-service", "callback"],
        "requires_ppr": True,
    },
    {
        "id": "MH-010", "category": "single_hop", "difficulty": "easy",
        "query": "Apa severity dari error ERR_AUTH_REDIRECT_MISMATCH?",
        "gold_keywords": ["HIGH", "403"],
        "requires_ppr": False,
    },
]


def load_golden_set(path: Optional[str] = None) -> List[Dict]:
    """Load golden set from JSON file or use built-in."""
    if path:
        with open(path) as f:
            data = json.load(f)
            return data.get("items", data) if isinstance(data, dict) else data
    return GOLDEN_SET


# ─── Metrics ─────────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    query_id: str
    query: str
    mode: str
    category: str
    requires_ppr: bool
    contexts: List[Dict[str, Any]]
    latency_ms: float
    hit_at_k: bool = False
    reciprocal_rank: float = 0.0
    keyword_recall: float = 0.0


def compute_hit_at_k(contexts: List[Dict], gold_keywords: List[str], k: int = 5) -> bool:
    for ctx in contexts[:k]:
        text = ctx.get("text", "").lower()
        if any(kw.lower() in text for kw in gold_keywords):
            return True
    return False


def compute_mrr(contexts: List[Dict], gold_keywords: List[str]) -> float:
    for idx, ctx in enumerate(contexts):
        text = ctx.get("text", "").lower()
        if any(kw.lower() in text for kw in gold_keywords):
            return 1.0 / (idx + 1)
    return 0.0


def compute_keyword_recall(contexts: List[Dict], gold_keywords: List[str]) -> float:
    if not gold_keywords:
        return 1.0
    all_text = " ".join(ctx.get("text", "") for ctx in contexts).lower()
    found = sum(1 for kw in gold_keywords if kw.lower() in all_text)
    return found / len(gold_keywords)


# ─── Eval Runner ─────────────────────────────────────────────────────────────

async def run_single_query(
    client: httpx.AsyncClient,
    golden: Dict[str, Any],
    mode: str,
) -> Optional[QueryResult]:
    start = time.monotonic()
    try:
        r = await client.post(f"{BASE_URL}/retrieve", json={
            "query": golden["query"],
            "top_k": 5,
            "mode": mode,
        })
        latency_ms = (time.monotonic() - start) * 1000

        if r.status_code != 200:
            print(f"  ✗ {golden['id']}: HTTP {r.status_code}")
            return None

        data = r.json()
        contexts = data.get("contexts", [])

        return QueryResult(
            query_id=golden["id"],
            query=golden["query"],
            mode=mode,
            category=golden.get("category", "unknown"),
            requires_ppr=golden.get("requires_ppr", False),
            contexts=contexts,
            latency_ms=latency_ms,
            hit_at_k=compute_hit_at_k(contexts, golden["gold_keywords"]),
            reciprocal_rank=compute_mrr(contexts, golden["gold_keywords"]),
            keyword_recall=compute_keyword_recall(contexts, golden["gold_keywords"]),
        )
    except Exception as e:
        print(f"  ✗ {golden['id']}: {e}")
        return None


def print_comparison_table(all_results: Dict[str, List[QueryResult]]):
    """Print 3-way comparison table with delta %."""
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)

    # Overall metrics
    header = f"{'Mode':<15} {'Hit@5':>8} {'MRR':>8} {'KW Recall':>10} {'Latency':>10}"
    print(header)
    print("-" * len(header))

    baseline_hit = baseline_mrr = None
    mode_stats = {}

    for m, results in all_results.items():
        if not results:
            print(f"{m:<15} {'N/A':>8}")
            continue

        total = len(results)
        hit = sum(1 for r in results if r.hit_at_k) / total
        mrr = sum(r.reciprocal_rank for r in results) / total
        kw = sum(r.keyword_recall for r in results) / total
        lat = sum(r.latency_ms for r in results) / total

        mode_stats[m] = {"hit": hit, "mrr": mrr, "kw": kw, "lat": lat}

        if baseline_hit is None:
            baseline_hit, baseline_mrr = hit, mrr

        delta = ""
        if baseline_hit is not None and m != list(all_results.keys())[0]:
            delta = f"  Δ{(hit - baseline_hit):+.1%}"
        print(f"{m:<15} {hit:>7.1%} {mrr:>8.3f} {kw:>10.3f} {lat:>9.0f}ms{delta}")

    # PPR-specific analysis: queries where requires_ppr=True
    ppr_modes = ["hybrid", "linear_rag"]
    if all(m in all_results for m in ppr_modes):
        hybrid_results = all_results["hybrid"]
        ppr_results = all_results["linear_rag"]

        ppr_queries = [r for r in hybrid_results if r.requires_ppr]
        non_ppr_queries = [r for r in hybrid_results if not r.requires_ppr]

        if ppr_queries:
            print(f"\n--- PPR Value Analysis ---")
            print(f"{'Category':<25} {'Hybrid Hit':>12} {'PPR Hit':>12} {'PPR Gain':>10}")
            print("-" * 60)

            # Group by requires_ppr
            for label, queries in [("requires_ppr=True", ppr_queries), ("requires_ppr=False", non_ppr_queries)]:
                hybrid_hit = sum(1 for r in queries if r.hit_at_k) / len(queries) if queries else 0
                ppr_hit = sum(1 for r in ppr_results if r.query_id in {q.query_id for q in queries} and r.hit_at_k)
                ppr_hit = ppr_hit / len(queries) if queries else 0
                gain = ppr_hit - hybrid_hit
                print(f"{label:<25} {hybrid_hit:>11.1%} {ppr_hit:>11.1%} {gain:>+9.1%}")

    # Per-category breakdown
    categories = set(r.category for r in all_results.get("linear_rag", []))
    if categories and "hybrid" in all_results and "linear_rag" in all_results:
        print(f"\n--- Per-Category Breakdown ---")
        print(f"{'Category':<25} {'Hybrid Hit':>12} {'PPR Hit':>12} {'PPR Gain':>10}")
        print("-" * 60)

        for cat in sorted(categories):
            hybrid_cat = [r for r in all_results["hybrid"] if r.category == cat]
            ppr_cat = [r for r in all_results["linear_rag"] if r.category == cat]

            h_hit = sum(1 for r in hybrid_cat if r.hit_at_k) / len(hybrid_cat) if hybrid_cat else 0
            p_hit = sum(1 for r in ppr_cat if r.hit_at_k) / len(ppr_cat) if ppr_cat else 0
            gain = p_hit - h_hit
            marker = " ✓" if gain > 0 else (" =" if gain == 0 else " ✗")
            print(f"{cat:<25} {h_hit:>11.1%} {p_hit:>11.1%} {gain:>+9.1%}{marker}")


async def run_eval(mode: str = "all", samples: Optional[int] = None, golden_set_path: Optional[str] = None):
    """Run full evaluation with 3-way comparison."""
    print("=" * 60)
    print("LinearRag-RAG Evaluation Harness (3-Way Comparison)")
    print("=" * 60)

    if mode == "all":
        modes = ["bm25_only", "vector_only", "hybrid", "linear_rag"]
    else:
        modes = [mode]

    dataset = load_golden_set(golden_set_path)
    if samples:
        dataset = dataset[:samples]
    print(f"Golden set: {len(dataset)} queries, modes: {modes}")

    all_results: Dict[str, List[QueryResult]] = {}

    async with httpx.AsyncClient(timeout=60.0, headers=HEADERS) as client:
        try:
            r = await client.get(f"{BASE_URL}/health")
            if r.status_code != 200:
                print("Server not healthy!")
                return
        except Exception:
            print(f"Server not reachable at {BASE_URL}")
            return

        for m in modes:
            print(f"\n--- Mode: {m} ---")
            results = []

            for golden in dataset:
                print(f"  [{golden['id']}] {golden['query'][:55]}...")
                result = await run_single_query(client, golden, m)
                if result:
                    status = "✓" if result.hit_at_k else "✗"
                    ppr_flag = "⚡" if result.requires_ppr else " "
                    print(f"    {status}{ppr_flag} hit={result.hit_at_k}, mrr={result.reciprocal_rank:.2f}, "
                          f"kw={result.keyword_recall:.2f}, {result.latency_ms:.0f}ms, "
                          f"ctx={len(result.contexts)}")
                    results.append(result)

            all_results[m] = results

    # Summary per mode
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for m, results in all_results.items():
        if not results:
            print(f"\n{m}: No results")
            continue
        total = len(results)
        hits = sum(1 for r in results if r.hit_at_k)
        mrr = sum(r.reciprocal_rank for r in results) / total
        avg_kw = sum(r.keyword_recall for r in results) / total
        avg_lat = sum(r.latency_ms for r in results) / total

        print(f"\n{m}:")
        print(f"  Queries:     {total}")
        print(f"  Hit@5:       {hits}/{total} ({hits/total:.1%})")
        print(f"  MRR:         {mrr:.3f}")
        print(f"  KW Recall:   {avg_kw:.3f}")
        print(f"  Avg Latency: {avg_lat:.0f}ms")

    # 3-way comparison table
    if len(all_results) >= 2:
        print_comparison_table(all_results)

    # Save results
    export = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "golden_set_size": len(dataset),
        "golden_set_path": golden_set_path or "built-in",
        "modes": {},
    }
    for m, results in all_results.items():
        export["modes"][m] = {
            "queries": len(results),
            "hit_at_k": sum(1 for r in results if r.hit_at_k),
            "mrr": sum(r.reciprocal_rank for r in results) / len(results) if results else 0,
            "avg_kw_recall": sum(r.keyword_recall for r in results) / len(results) if results else 0,
            "avg_latency_ms": sum(r.latency_ms for r in results) / len(results) if results else 0,
            "details": [
                {
                    "id": r.query_id,
                    "query": r.query,
                    "category": r.category,
                    "requires_ppr": r.requires_ppr,
                    "hit": r.hit_at_k,
                    "mrr": r.reciprocal_rank,
                    "kw_recall": r.keyword_recall,
                    "latency_ms": round(r.latency_ms),
                    "contexts": len(r.contexts),
                }
                for r in results
            ],
        }

    export_path = Path("evals/eval_results.json")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"\nResults saved to {export_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    global BASE_URL
    parser = argparse.ArgumentParser(description="LinearRag-RAG Eval Harness (3-Way)")
    parser.add_argument("--mode", choices=["bm25_only", "hybrid", "linear_rag", "all"], default="all",
                        help="Retrieval mode to evaluate")
    parser.add_argument("--samples", type=int, default=None,
                        help="Number of golden set samples to evaluate")
    parser.add_argument("--golden-set", type=str, default=None,
                        help="Path to JSON golden set file")
    parser.add_argument("--url", default=BASE_URL,
                        help="LinearRag-RAG server URL")
    args = parser.parse_args()

    BASE_URL = args.url

    asyncio.run(run_eval(mode=args.mode, samples=args.samples, golden_set_path=args.golden_set))


if __name__ == "__main__":
    main()
