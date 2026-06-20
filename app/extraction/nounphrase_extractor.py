import spacy
import logging
from typing import List, Optional
from app.extraction.contracts import ExtractorContract, Chunk, ExtractionCandidate

logger = logging.getLogger(__name__)


class NounPhraseExtractor(ExtractorContract):
    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning("spaCy model '%s' not found, falling back to en_core_web_sm", model_name)
            self.nlp = spacy.load("en_core_web_sm")

    def extract_terms(
        self,
        chunks: List[Chunk],
        labels: List[str],
        threshold: float = 0.5,
        language_hint: Optional[str] = None,
    ) -> List[ExtractionCandidate]:
        candidates: List[ExtractionCandidate] = []

        for chunk in chunks:
            doc = self.nlp(chunk.content)
            for np in doc.noun_chunks:
                text = np.text.strip()
                if len(text) < 2:
                    continue
                if all(t.is_stop or t.is_punct or t.is_space for t in np):
                    continue

                candidates.append(
                    ExtractionCandidate(
                        chunk_id=chunk.chunk_id,
                        text=text,
                        label="concept",
                        start_offset=np.start_char,
                        end_offset=np.end_char,
                        confidence=1.0,
                    )
                )
        return candidates
