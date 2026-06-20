import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock
from app.retrieval.linear_rag import LinearRagRetriever

import asyncio

def test_run_personalized_pagerank_returns_documents():
    asyncio.run(_test_run_personalized_pagerank_returns_documents())

async def _test_run_personalized_pagerank_returns_documents():
    repo_mock = AsyncMock()
    embedder_mock = MagicMock()
    
    # We have chunk 1 and chunk 2. 
    # Chunk 1 belongs to Doc 10
    # Chunk 2 belongs to Doc 20
    chunk_weights = {1: 0.8, 2: 0.6}
    actived_entities = {100: 0.5}
    
    # Mock mentions: Chunk 1 mentions Entity 100
    repo_mock.get_mentions_for_chunks.return_value = {1: [100]}
    
    # Mock chunks mapping to documents
    repo_mock.get_chunks_by_ids.return_value = {
        1: {"document_id": 10, "text": "chunk 1 text"},
        2: {"document_id": 20, "text": "chunk 2 text"},
    }
    
    retriever = LinearRagRetriever(repo=repo_mock, embedding_model=embedder_mock)
    
    # Run Tri-Graph PPR
    final_scores, vcount, ecount = await retriever._run_personalized_pagerank(
        chunk_weights=chunk_weights,
        actived_entities=actived_entities,
        top_k=2
    )
    
    # The output should be Document IDs, not Chunk IDs!
    # Because Entity 100 is connected to Chunk 1, and Chunk 1 is connected to Doc 10.
    # Doc 10 should be retrieved.
    assert len(final_scores) <= 2
    
    # Check if the returned IDs are 10 and 20
    returned_ids = [doc_id for doc_id, score in final_scores]
    assert 10 in returned_ids
    assert 20 in returned_ids
    
    # Because Chunk 1 has a mention to Entity 100 (which is active), 
    # Document 10 should ideally get a higher PPR score than Document 20.
    doc10_score = next(score for doc_id, score in final_scores if doc_id == 10)
    doc20_score = next(score for doc_id, score in final_scores if doc_id == 20)
    
    assert doc10_score > doc20_score
