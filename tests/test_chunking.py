import pytest
from app.chunking.recursive import RecursiveChunker, estimate_tokens
from app.chunking.contracts import ChunkDocumentInput, ChunkingOptions


def test_estimate_tokens():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 4) == 1
    assert estimate_tokens("a" * 8) == 2


def test_chunker_basic():
    chunker = RecursiveChunker()
    result = chunker.chunk_document(
        ChunkDocumentInput(content="Hello world. " * 200, source_type="text"),
        ChunkingOptions(target_tokens=50, overlap_tokens=10),
    )
    assert len(result) > 1
    for chunk in result:
        assert chunk.content
        assert chunk.start_offset >= 0
        assert chunk.end_offset > chunk.start_offset


def test_chunker_empty():
    chunker = RecursiveChunker()
    result = chunker.chunk_document(
        ChunkDocumentInput(content="", source_type="text"),
        ChunkingOptions(target_tokens=512, overlap_tokens=96),
    )
    assert len(result) == 0


def test_chunker_small_content():
    chunker = RecursiveChunker()
    result = chunker.chunk_document(
        ChunkDocumentInput(content="Hello world", source_type="text"),
        ChunkingOptions(target_tokens=512, overlap_tokens=96),
    )
    assert len(result) == 1
    assert result[0].content == "Hello world"
