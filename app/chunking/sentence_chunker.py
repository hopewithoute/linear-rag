import re
import math
from typing import List
from app.chunking.contracts import ChunkerContract, ChunkDocumentInput, ChunkingOptions, DocumentChunk

def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4.0))

class SentenceChunker(ChunkerContract):
    """
    Splits text strictly into sentences using punctuation (. ! ? \\n)
    as required by the LinearRAG Tri-Graph architecture.
    """
    def chunk_document(self, input_data: ChunkDocumentInput, options: ChunkingOptions) -> List[DocumentChunk]:
        if not input_data.content:
            return []
            
        text = input_data.content
        
        # Split by punctuation: '.', '!', '?', or '\n'
        # The regex keeps the punctuation with the sentence
        pattern = r'([^.!?\n]+[.!?\n]+)'
        
        # If there's leftover text without punctuation, it will be in the parts as well
        parts = re.split(pattern, text)
        
        chunks = []
        offset = 0
        chunk_idx = 0
        
        for part in parts:
            if not part:
                continue
                
            stripped = part.strip()
            if not stripped:
                offset += len(part)
                continue
                
            start = text.find(part, offset)
            if start == -1:
                start = offset
                
            end = start + len(part)
            offset = end
            
            chunks.append(
                DocumentChunk(
                    chunk_index=chunk_idx,
                    content=part,
                    start_offset=start,
                    end_offset=end,
                    estimated_tokens=estimate_tokens(part)
                )
            )
            chunk_idx += 1
            
        return chunks
