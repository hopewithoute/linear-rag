import pytest
from app.chunking.contracts import ChunkDocumentInput, ChunkingOptions
from app.chunking.sentence_chunker import SentenceChunker

def test_sentence_chunker_splits_by_punctuation():
    chunker = SentenceChunker()
    
    text = "Hello world. How are you today? I am fine! This is a new line.\nAnother line."
    options = ChunkingOptions(target_tokens=512, overlap_tokens=0)
    
    chunks = chunker.chunk_document(
        ChunkDocumentInput(content=text, source_type="text"), 
        options
    )
    
    # We expect the text to be split strictly by punctuation
    # "Hello world.", "How are you today?", "I am fine!", "This is a new line.", "Another line."
    assert len(chunks) == 5
    assert chunks[0].content.strip() == "Hello world."
    assert chunks[1].content.strip() == "How are you today?"
    assert chunks[2].content.strip() == "I am fine!"
    assert chunks[3].content.strip() == "This is a new line."
    assert chunks[4].content.strip() == "Another line."
