import re
from typing import List, Optional
from app.extraction.contracts import ExtractorContract, Chunk, ExtractionCandidate

PATTERNS = {
    "ERROR_CODE": re.compile(r"\b[A-Z][A-Z0-9_]{2,30}\b"),
    "VERSION": re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b"),
    "ENDPOINT": re.compile(r"(?:GET|POST|PUT|DELETE|PATCH)\s+/[\w/\-{}]+|/api/v\d+/[\w/\-]+"),
    "HTTP_STATUS": re.compile(r"\b[1-5]\d{2}\b"),
    "UUID": re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE),
    "TENANT_ID": re.compile(r"\btenant[_\-]?\d+\b", re.IGNORECASE),
    "DATE": re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    "FILE_PATH": re.compile(r"(?:/[\w.\-]+){2,}"),
}

DOMAIN_STOPWORDS = frozenset([
    "the", "this", "that", "with", "from", "have", "will", "been", "were", "are",
    "was", "for", "and", "but", "not", "can", "may", "use", "using", "used",
])


class RegexExtractor(ExtractorContract):
    def extract_terms(
        self,
        chunks: List[Chunk],
        labels: List[str],
        threshold: float = 0.5,
        language_hint: Optional[str] = None,
    ) -> List[ExtractionCandidate]:
        candidates: List[ExtractionCandidate] = []

        for chunk in chunks:
            for label, pattern in PATTERNS.items():
                for match in pattern.finditer(chunk.content):
                    text = match.group().strip()
                    if len(text) < 2:
                        continue
                    if text.lower() in DOMAIN_STOPWORDS:
                        continue

                    candidates.append(
                        ExtractionCandidate(
                            chunk_id=chunk.chunk_id,
                            text=text,
                            label=label,
                            start_offset=match.start(),
                            end_offset=match.end(),
                            confidence=0.9,
                        )
                    )

        return candidates
