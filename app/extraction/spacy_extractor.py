import spacy
from typing import List, Optional
from app.extraction.contracts import ExtractorContract, Chunk, ExtractionCandidate

# Module-level model cache
_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            _nlp = spacy.load("en_core_web_sm")
    return _nlp

class SpacyExtractor(ExtractorContract):
    def extract_terms(
        self,
        chunks: List[Chunk],
        labels: List[str],
        threshold: float = 0.0,
        language_hint: Optional[str] = None,
    ) -> List[ExtractionCandidate]:
        nlp = get_nlp()
        candidates: List[ExtractionCandidate] = []
        
        for chunk in chunks:
            if not chunk.content:
                continue
                
            doc = nlp(chunk.content)
            
            for ent in doc.ents:
                # Based on the paper, we want named entities. 
                # We can filter out things like CARDINAL and ORDINAL, or keep them all.
                # LinearRAG focuses on structural entities, let's filter purely quantitative entities
                if ent.label_ in ("ORDINAL", "CARDINAL", "QUANTITY", "PERCENT", "MONEY"):
                    continue
                    
                start = ent.start_char
                end = ent.end_char
                text = ent.text
                label = ent.label_
                
                if start < 0 or end <= start or not text:
                    continue

                candidates.append(
                    ExtractionCandidate(
                        chunk_id=chunk.chunk_id,
                        text=text,
                        label=label,
                        start_offset=start,
                        end_offset=end,
                        confidence=1.0,  # spaCy doesn't provide confidence scores by default
                    )
                )

        return candidates
