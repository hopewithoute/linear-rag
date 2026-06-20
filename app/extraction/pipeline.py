from typing import List, Optional
from app.extraction.contracts import ExtractorContract, Chunk, ExtractionCandidate


class TermExtractionPipeline(ExtractorContract):
    def __init__(self, extractors: list[ExtractorContract]):
        self.extractors = extractors

    def extract_terms(
        self,
        chunks: List[Chunk],
        labels: List[str],
        threshold: float = 0.5,
        language_hint: Optional[str] = None,
    ) -> List[ExtractionCandidate]:
        candidates: List[ExtractionCandidate] = []
        for ext in self.extractors:
            candidates.extend(ext.extract_terms(chunks, labels, threshold, language_hint))

        normalized: List[ExtractionCandidate] = []
        seen: set[tuple[str, str, str, int, int]] = set()

        for cand in candidates:
            cand.text = cand.text.strip()
            if not cand.text:
                continue
            norm_text = cand.text.lower()
            cand.normalized_text = norm_text
            key = (cand.chunk_id, norm_text, cand.label, cand.start_offset, cand.end_offset)
            if key not in seen:
                seen.add(key)
                normalized.append(cand)

        return normalized
