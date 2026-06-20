"""Stop-entity list for context drift prevention.

Common entities that are too generic and should be filtered or penalized
during graph construction and retrieval.
"""
from typing import Optional

# Default stop-entities (too common, cause context drift)
DEFAULT_STOP_ENTITIES = frozenset([
    # General
    "user", "data", "system", "error", "problem", "issue",
    "result", "solution", "method", "process", "service",
    "application", "platform", "tool", "feature", "function",
    
    # Tech generic
    "api", "url", "http", "https", "web", "server", "client",
    "database", "table", "column", "row", "query",
    "request", "response", "header", "body",
    
    # People/Places (too common)
    "person", "people", "organization", "company", "country",
    "city", "state", "region", "location",
    
    # Time (too common)
    "date", "time", "year", "month", "day", "hour", "minute",
    
    # Indonesian (common)
    "sistem", "data", "aplikasi", "layanan", "masalah",
    "solusi", "proses", "metode", "fitur", "fungsi",
])


def is_stop_entity(entity: str, stop_list: Optional[frozenset] = None) -> bool:
    """Check if entity is a stop-entity that should be filtered."""
    stop = stop_list or DEFAULT_STOP_ENTITIES
    normalized = entity.lower().strip()
    return normalized in stop


def get_entity_penalty(entity: str, stop_list: Optional[frozenset] = None) -> float:
    """Get penalty factor for entity (0.0 = full stop, 1.0 = no penalty)."""
    if is_stop_entity(entity, stop_list):
        return 0.0  # Full stop
    return 1.0  # No penalty

