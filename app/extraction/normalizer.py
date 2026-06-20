import re
import unicodedata
from typing import Dict, Tuple

_ENTITY_LENGTH_EXCEPTIONS = frozenset(["ERR", "OOM", "SSO", "RBAC", "API", "DNS", "HTTP", "SSL", "TLS"])


def normalize_entity(text: str) -> str:
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


def canonical_entity(text: str, alias_table: Dict[str, str]) -> Tuple[str, str]:
    normalized = normalize_entity(text)
    if normalized in alias_table:
        return alias_table[normalized], normalized
    return normalized, normalized
