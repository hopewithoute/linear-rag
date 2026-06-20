import os
import httpx
from typing import List, Optional
from app.extraction.contracts import ExtractorContract, Chunk, ExtractionCandidate

class RemoteExtractor(ExtractorContract):
    def __init__(self, url: str, token: str = "dev-secret-key"):
        self.url = url
        self.token = token
        
    def extract_terms(self, chunks: List[Chunk], labels: List[str], threshold: float = 0.5, language_hint: Optional[str] = None) -> List[ExtractionCandidate]:
        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {
            "chunks": [{"chunkId": c.chunk_id, "content": c.content} for c in chunks],
            "labels": labels,
            "threshold": threshold,
            "languageHint": language_hint
        }
        
        try:
            with httpx.Client(timeout=600.0) as client:
                response = client.post(f"{self.url.rstrip('/')}/extract", json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                result = []
                for cand in data.get("candidates", []):
                    result.append(ExtractionCandidate(
                        chunk_id=cand.get("chunkId", ""),
                        text=cand.get("text", ""),
                        label=cand.get("label", ""),
                        start_offset=cand.get("startOffset", 0),
                        end_offset=cand.get("endOffset", 0),
                        confidence=cand.get("confidence", 0.0)
                    ))
                return result
        except Exception as exc:
            raise RuntimeError(f"HTTP extractor inference failed: {exc}") from exc
