from typing import List, Literal
from pydantic import BaseModel


class ChunkDocumentInput(BaseModel):
    document_id: str | None = None
    content: str
    source_type: Literal["text", "markdown"] = "text"


class ChunkingOptions(BaseModel):
    target_tokens: int = 512
    overlap_tokens: int = 96
    min_chunk_tokens: int = 80


class DocumentChunk(BaseModel):
    chunk_index: int
    content: str
    start_offset: int
    end_offset: int
    estimated_tokens: int


class ChunkerContract:
    def chunk_document(self, input_data: ChunkDocumentInput, options: ChunkingOptions) -> List[DocumentChunk]:
        raise NotImplementedError
