import pytest
from app.extraction.contracts import Chunk, ExtractionCandidate
from app.extraction.spacy_extractor import SpacyExtractor

def test_spacy_extractor_extracts_entities():
    # We pass a simple sentence with known entities
    extractor = SpacyExtractor()
    chunks = [
        Chunk(chunk_id="1", content="Apple is looking at buying U.K. startup for $1 billion")
    ]
    
    # We ignore labels and threshold for SpacyExtractor
    candidates = extractor.extract_terms(chunks, labels=[], threshold=0.0)
    
    # We expect Apple (ORG) and U.K. (GPE) to be extracted
    texts = [c.text for c in candidates]
    
    assert "Apple" in texts
    assert "U.K." in texts
    
    # Verify candidate properties
    apple_cand = next(c for c in candidates if c.text == "Apple")
    assert apple_cand.label == "ORG"
    assert apple_cand.chunk_id == "1"
