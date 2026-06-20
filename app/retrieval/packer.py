from typing import List, Dict, Any


def pack_contexts(
    ranked_chunks: List[Dict[str, Any]],
    top_k: int = 12,
    max_per_doc: int = 4,
) -> List[Dict[str, Any]]:
    seen_texts: set[str] = set()
    doc_counts: Dict[str, int] = {}
    packed: List[Dict[str, Any]] = []

    for chunk in ranked_chunks:
        if len(packed) >= top_k:
            break

        text = chunk.get("text", "")
        if text in seen_texts:
            continue
        seen_texts.add(text)

        doc_id = str(chunk.get("document_id", ""))
        if doc_counts.get(doc_id, 0) >= max_per_doc:
            continue
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

        packed.append({
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk.get("document_id"),
            "score": chunk.get("score", 0),
            "text": text,
            "evidence_path": chunk.get("evidence_path", []),
            "rerank_boost": chunk.get("rerank_boost"),
        })

    return packed
