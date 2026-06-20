import re
import unicodedata
from typing import Dict, Tuple, List, Optional
from difflib import SequenceMatcher

# Code-like patterns that should preserve case
_ENTITY_LENGTH_EXCEPTIONS = frozenset([
    "ERR", "OOM", "SSO", "RBAC", "API", "DNS", "HTTP", "SSL", "TLS",
    "JSON", "YAML", "REST", "gRPC", "SQL", "CSS", "HTML",
])

# Common entity aliases (normalized -> canonical)
DEFAULT_ALIASES: Dict[str, str] = {
    "openai": "OpenAI",
    "open ai": "OpenAI",
    "openai inc": "OpenAI",
    "chat gpt": "ChatGPT",
    "chatgpt": "ChatGPT",
    "microsoft": "Microsoft",
    "ms": "Microsoft",
    "google": "Google",
    "aws": "AWS",
    "amazon web services": "AWS",
}


def normalize_entity(text: str) -> str:
    """Basic entity normalization."""
    text = text.strip()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(".,;:!?)(")

    is_code_like = (
        text.upper() == text
        and len(text) >= 3
        and any(c.isdigit() for c in text)
    ) or text in _ENTITY_LENGTH_EXCEPTIONS

    if not is_code_like:
        text = text.lower()

    return text


def canonical_entity(text: str, alias_table: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
    """Resolve entity to canonical form using alias table."""
    normalized = normalize_entity(text)
    table = alias_table or DEFAULT_ALIASES

    if normalized in table:
        return table[normalized], normalized
    return normalized, normalized


def fuzzy_match_entity(
    text: str,
    existing_entities: List[str],
    threshold: float = 0.85,
) -> Optional[str]:
    """Find best matching entity using fuzzy string matching."""
    normalized = normalize_entity(text)

    best_match = None
    best_score = 0.0

    for entity in existing_entities:
        entity_normalized = normalize_entity(entity)

        # Exact match
        if normalized == entity_normalized:
            return entity

        # Fuzzy match
        score = SequenceMatcher(None, normalized, entity_normalized).ratio()
        if score > best_score and score >= threshold:
            best_score = score
            best_match = entity

    return best_match


def compute_entity_similarity(text1: str, text2: str) -> float:
    """Compute similarity between two entity texts."""
    n1 = normalize_entity(text1)
    n2 = normalize_entity(text2)

    if n1 == n2:
        return 1.0

    # Prefix match (e.g., "acme" matches "acmeid")
    if n1.startswith(n2) or n2.startswith(n1):
        shorter = min(len(n1), len(n2))
        longer = max(len(n1), len(n2))
        if shorter / longer > 0.6:
            return 0.9

    return SequenceMatcher(None, n1, n2).ratio()


def build_alias_table_from_entities(
    entities: List[Dict[str, str]],
    similarity_threshold: float = 0.85,
) -> Dict[str, str]:
    """Build alias table by clustering similar entity names."""
    canonical_map: Dict[str, str] = {}

    for entity in entities:
        name = entity.get("canonical_name", "")
        if not name:
            continue

        normalized = normalize_entity(name)

        # Find existing canonical that matches
        matched = None
        for existing_norm, canonical in canonical_map.items():
            if compute_entity_similarity(normalized, existing_norm) >= similarity_threshold:
                matched = canonical
                break

        if matched:
            canonical_map[normalized] = matched
        else:
            canonical_map[normalized] = name

    return canonical_map
