from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Module-level model cache
_cross_encoder_cache = {}


@dataclass
class RerankerConfig:
    mode: str = "heuristic"  # none, heuristic, cross_encoder
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    boost_exact_match: float = 0.1
    boost_entity_overlap: float = 0.05
    boost_short_chunk: float = 0.02
    penalize_long_chunk: float = 0.02
    short_chunk_threshold: int = 500
    long_chunk_threshold: int = 2000


def rerank_contexts(
    contexts: List[Dict[str, Any]],
    query: str,
    config: Optional[RerankerConfig] = None,
) -> List[Dict[str, Any]]:
    """Rerank contexts based on query."""
    if not contexts:
        return contexts

    cfg = config or RerankerConfig()

    if cfg.mode == "none":
        return contexts

    if cfg.mode == "heuristic":
        return _heuristic_rerank(contexts, query, cfg)

    if cfg.mode == "cross_encoder":
        return _cross_encoder_rerank(contexts, query, cfg)

    return contexts


def _get_cross_encoder(model_name: str):
    """Get or create cached cross-encoder model."""
    if model_name not in _cross_encoder_cache:
        from sentence_transformers import CrossEncoder
        _cross_encoder_cache[model_name] = CrossEncoder(model_name)
    return _cross_encoder_cache[model_name]


def _heuristic_rerank(
    contexts: List[Dict[str, Any]],
    query: str,
    config: RerankerConfig,
) -> List[Dict[str, Any]]:
    """Rule-based reranking tanpa ML."""
    query_lower = query.lower()
    query_words = set(query_lower.split())

    for ctx in contexts:
        boost = 0.0
        text = ctx.get("text", "").lower()

        # 1. Boost exact match (highest priority)
        if query_lower in text:
            boost += config.boost_exact_match

        # 2. Boost word overlap (high priority)
        text_words = set(text.split())
        overlap = len(query_words & text_words)
        if overlap > 0:
            boost += overlap * config.boost_entity_overlap

        # 3. Boost title match (if chunk starts with "Title:")
        if text.startswith("title:"):
            title = text.split("\n")[0].replace("title:", "").strip()
            title_words = set(title.split())
            title_overlap = len(query_words & title_words)
            if title_overlap > 0:
                boost += title_overlap * 0.15  # Higher boost for title match

        # 4. Boost short chunks (more focused)
        if len(ctx.get("text", "")) < config.short_chunk_threshold:
            boost += config.boost_short_chunk

        # 5. Penalize very long chunks (less focused)
        if len(ctx.get("text", "")) > config.long_chunk_threshold:
            boost -= config.penalize_long_chunk

        ctx["rerank_boost"] = boost
        ctx["score"] = round(ctx.get("score", 0) + boost, 4)

    return sorted(contexts, key=lambda x: x.get("score", 0), reverse=True)


def _cross_encoder_rerank(
    contexts: List[Dict[str, Any]],
    query: str,
    config: RerankerConfig,
) -> List[Dict[str, Any]]:
    """Cross-encoder reranking."""
    try:
        reranker = _get_cross_encoder(config.cross_encoder_model)

        pairs = [(query, ctx.get("text", "")) for ctx in contexts]
        ce_scores = reranker.predict(pairs)

        # Normalize CE scores to [0, 1] range
        min_score = min(ce_scores)
        max_score = max(ce_scores)
        score_range = max_score - min_score if max_score != min_score else 1.0

        for ctx, ce_score in zip(contexts, ce_scores):
            normalized_score = (ce_score - min_score) / score_range
            ctx["rerank_score"] = float(ce_score)
            # Use CE score as ranking signal but keep original score structure
            ctx["score"] = round(ctx.get("score", 0) * 0.3 + normalized_score * 0.7, 4)

        return sorted(contexts, key=lambda x: x.get("score", 0), reverse=True)

    except ImportError:
        import logging
        logger = logging.getLogger("linear_rag-rag.reranker")
        logger.warning("sentence-transformers not installed, falling back to heuristic")
        return _heuristic_rerank(contexts, query, config)
    except Exception as e:
        import logging
        logger = logging.getLogger("linear_rag-rag.reranker")
        logger.warning(f"Cross-encoder rerank failed: {e}, falling back to heuristic")
        return _heuristic_rerank(contexts, query, config)
