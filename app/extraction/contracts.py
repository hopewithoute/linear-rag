from typing import List, Optional
from pydantic import BaseModel


class Chunk(BaseModel):
    chunk_id: str
    content: str


class ExtractionCandidate(BaseModel):
    chunk_id: str
    text: str
    label: str
    start_offset: int
    end_offset: int
    confidence: float
    normalized_text: Optional[str] = None


class ExtractorContract:
    def extract_terms(
        self,
        chunks: List[Chunk],
        labels: List[str],
        threshold: float = 0.5,
        language_hint: Optional[str] = None,
    ) -> List[ExtractionCandidate]:
        raise NotImplementedError
