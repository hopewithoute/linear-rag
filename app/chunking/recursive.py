import math
from typing import List
from app.chunking.contracts import ChunkerContract, ChunkDocumentInput, ChunkingOptions, DocumentChunk


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4.0))


SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]


def recursively_split(text: str, target_tokens: int, offset: int = 0) -> List[tuple[str, int, int]]:
    if estimate_tokens(text) <= target_tokens:
        return [(text, offset, offset + len(text))]

    for separator in SEPARATORS:
        if not separator:
            return [(char, offset + idx, offset + idx + 1) for idx, char in enumerate(text)]

        parts = text.split(separator)
        # Prevent infinite recursion if the separator is just at the very end
        if len(parts) > 1 and not (len(parts) == 2 and parts[1] == ""):
            result = []
            current_offset = offset
            for i, part in enumerate(parts):
                chunk_text = part + (separator if i < len(parts) - 1 else "")
                if chunk_text:
                    result.extend(recursively_split(chunk_text, target_tokens, current_offset))
                current_offset += len(chunk_text)
            return result

    return [(text, offset, offset + len(text))]


def pack_chunks(splits: list[tuple[str, int, int]], options: ChunkingOptions) -> List[tuple[str, int, int]]:
    chunks = []
    current_parts: list[tuple[str, int, int]] = []

    for part_text, part_start, part_end in splits:
        current_text = "".join(p[0] for p in current_parts)
        current_tokens = estimate_tokens(current_text)
        part_tokens = estimate_tokens(part_text)

        if not current_parts or current_tokens + part_tokens <= options.target_tokens:
            current_parts.append((part_text, part_start, part_end))
        else:
            full_text = "".join(p[0] for p in current_parts)
            chunks.append((full_text, current_parts[0][1], current_parts[-1][2]))
            overlap_tokens = 0
            overlap_parts: list[tuple[str, int, int]] = []
            for p in reversed(current_parts):
                t = estimate_tokens(p[0])
                if overlap_tokens + t <= options.overlap_tokens:
                    overlap_parts.insert(0, p)
                    overlap_tokens += t
                else:
                    break
            current_parts = overlap_parts + [(part_text, part_start, part_end)]

    if current_parts:
        full_text = "".join(p[0] for p in current_parts)
        if full_text.strip():
            chunks.append((full_text, current_parts[0][1], current_parts[-1][2]))

    return chunks


class RecursiveChunker(ChunkerContract):
    def chunk_document(self, input_data: ChunkDocumentInput, options: ChunkingOptions) -> List[DocumentChunk]:
        if not input_data.content:
            return []

        splits = recursively_split(input_data.content, options.target_tokens)
        packed = pack_chunks(splits, options)

        return [
            DocumentChunk(
                chunk_index=idx,
                content=text,
                start_offset=start,
                end_offset=end,
                estimated_tokens=estimate_tokens(text),
            )
            for idx, (text, start, end) in enumerate(packed)
        ]
