import time
import math
import numpy as np
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from sqlalchemy import select, func

import spacy
from functools import lru_cache

from app.storage.search_repository import SearchRepository
from app.storage.models import Entity, Chunk, Mention
from app.embedding.contracts import EmbeddingContract
from app.settings import get_settings

@dataclass
class RetrievalResult:
    query: str
    retrieval_mode: str
    contexts: List[Dict[str, Any]]
    debug: Dict[str, Any]

_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp

_embedding_cache = {}

class LinearRagRetriever:
    def __init__(self, repo: SearchRepository, embedding_model: EmbeddingContract):
        self.repo = repo
        self.embedding_model = embedding_model
        self.settings = get_settings()
        self.nlp = get_nlp()

    def _get_single_embedding(self, text: str):
        if text not in _embedding_cache:
            _embedding_cache[text] = self.embedding_model.create_embeddings([text])[0]
        return _embedding_cache[text]

    def extract_entities(self, text: str) -> List[str]:
        doc = self.nlp(text)
        entities = set()
        for ent in doc.ents:
            if ent.label_ not in ("ORDINAL", "CARDINAL"):
                entities.add(ent.text.lower())
        return list(entities)

    async def get_seed_entities(self, query: str, query_embedding: List[float]):
        """Find seed entities in DB matching the query NER, and compute initial scores."""
        query_ents = self.extract_entities(query)
        if not query_ents:
            return [], [], [], []
        from sqlalchemy import or_, literal
        conditions = []
        for q_ent in query_ents:
            # Match if query entity is substring of DB entity
            conditions.append(func.lower(Entity.canonical_name).ilike(f"%{q_ent}%"))
            # Match if DB entity is substring of query entity
            conditions.append(literal(q_ent).ilike(func.concat("%", func.lower(Entity.canonical_name), "%")))
            
        stmt = select(Entity).where(or_(*conditions))
        result = await self.repo.session.execute(stmt)
        db_entities = result.scalars().all()
        
        if not db_entities:
            return [], [], [], []
            
        # Encode canonical names on the fly

        
        entity_texts = [e.canonical_name for e in db_entities]
        
        # Use locally cached embeddings for each entity text, batching uncached
        uncached_texts = [t for t in entity_texts if t not in _embedding_cache]
        if uncached_texts:
            new_embs = self.embedding_model.create_embeddings(uncached_texts)
            for t, emb in zip(uncached_texts, new_embs):
                if len(_embedding_cache) > 10000:
                    _embedding_cache.pop(next(iter(_embedding_cache)))
                _embedding_cache[t] = emb
                
        entity_embeddings = [_embedding_cache[t] for t in entity_texts]
        
        scores = []
        q_emb = np.array(query_embedding)
        for emb in entity_embeddings:
            score = np.dot(np.array(emb), q_emb)
            scores.append(float(score))
            
        entity_ids = [e.entity_id for e in db_entities]
        
        return entity_ids, entity_texts, entity_embeddings, scores

    async def _expand_active_entities(self, query_embedding: List[float], seed_entity_ids: List[int], seed_entity_scores: List[float]) -> tuple[Dict[int, tuple[float, int]], Set[int]]:
        actived_entities = {}
        iteration_threshold = 0.5
        max_iterations = getattr(self.settings, "MAX_ITERATIONS", 3)
        top_k_sentence = 1
        
        for eid, score in zip(seed_entity_ids, seed_entity_scores):
            actived_entities[eid] = (score, 0)
            
        current_entities = {eid: (score, 0) for eid, score in zip(seed_entity_ids, seed_entity_scores)}
        used_chunk_ids = set()
        iteration = 1
        
        while len(current_entities) > 0 and iteration < max_iterations:
            new_entities = {}
            active_eids = [eid for eid, (score, _) in current_entities.items() if score >= iteration_threshold]
            if not active_eids:
                break
                
            chunks_map = await self.repo.get_chunks_for_entities(active_eids)
            
            for eid in active_eids:
                score, _ = current_entities[eid]
                available_chunks = [cid for cid in chunks_map.get(eid, []) if cid not in used_chunk_ids]
                if not available_chunks:
                    continue
                    
                chunk_embeddings = await self.repo.get_chunk_embeddings(available_chunks)
                if not chunk_embeddings:
                    continue
                    
                similarities = []
                for cid in available_chunks:
                    emb = chunk_embeddings.get(cid)
                    if emb:
                        sim = sum(a * b for a, b in zip(query_embedding, emb))
                        similarities.append((sim, cid))
                
                if not similarities:
                    continue
                    
                similarities.sort(reverse=True, key=lambda x: x[0])
                top_chunks = similarities[:top_k_sentence]
                
                for chunk_sim, top_cid in top_chunks:
                    used_chunk_ids.add(top_cid)
                    eids_in_chunk = await self.repo.get_mentions_for_chunks([top_cid])
                    eids_in_chunk = eids_in_chunk.get(top_cid, [])
                    
                    for next_eid in eids_in_chunk:
                        next_score = score * chunk_sim
                        if next_score >= iteration_threshold:
                            if next_eid not in new_entities:
                                new_entities[next_eid] = [0.0, iteration + 1]
                            new_entities[next_eid][0] += next_score

            formatted_new = {eid: (score, tier) for eid, (score, tier) in new_entities.items()}
            
            for eid, (score, tier) in formatted_new.items():
                if eid in actived_entities:
                    old_score, old_tier = actived_entities[eid]
                    actived_entities[eid] = (old_score + score, tier)
                else:
                    actived_entities[eid] = (score, tier)
                    
            current_entities = formatted_new
            iteration += 1
            
        return actived_entities, used_chunk_ids

    async def _run_personalized_pagerank(self, document_dpr_scores: Dict[int, float], chunk_cids: Set[int], actived_entities: Dict[int, tuple[float, int]], top_k: int) -> tuple[List[tuple[int, float]], int, int]:
        all_dids = list(document_dpr_scores.keys())
        all_cids = list(chunk_cids)
        all_eids = list(actived_entities.keys())
        
        if not all_dids or not all_eids:
            return [(did, float(document_dpr_scores[did])) for did in sorted(all_dids, key=lambda x: document_dpr_scores[x], reverse=True)[:top_k]], 0, 0
            
        all_cids_mentions_map = await self.repo.get_mentions_for_chunks(all_cids) if all_cids else {}
        
        # We need Document <-> Entity edges. We calculate this by aggregating Chunk <-> Entity for chunks in the documents.
        doc_chunks_map = await self.repo.get_chunks_by_document_ids(all_dids)
        doc_entity_counts = {}
        doc_total_counts = {}
        
        for did in all_dids:
            chunks = doc_chunks_map.get(did, [])
            cids_in_doc = [c["chunk_id"] for c in chunks]
            ent_counts = {}
            total_count = 0
            
            mentions_in_doc = await self.repo.get_mentions_for_chunks(cids_in_doc)
            for cid_doc, eids in mentions_in_doc.items():
                for eid in eids:
                    if eid in all_eids:
                        ent_counts[eid] = ent_counts.get(eid, 0) + 1
                        total_count += 1
                        
            doc_entity_counts[did] = ent_counts
            doc_total_counts[did] = total_count
        
        node_to_idx = {}
        idx_to_node = {}
        idx = 0
        
        for did in all_dids:
            node_id = f"d_{did}"
            node_to_idx[node_id] = idx
            idx_to_node[idx] = node_id
            idx += 1
            
        for cid in all_cids:
            node_id = f"c_{cid}"
            node_to_idx[node_id] = idx
            idx_to_node[idx] = node_id
            idx += 1
            
        for eid in all_eids:
            node_id = f"e_{eid}"
            node_to_idx[node_id] = idx
            idx_to_node[idx] = node_id
            idx += 1
            
        num_nodes = idx
        W = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        
        edge_count = 0
        
        # 1. Chunk <-> Entity edges (weight = 1.0)
        for cid in all_cids:
            c_idx = node_to_idx[f"c_{cid}"]
            eids_in_chunk = [e for e in all_cids_mentions_map.get(cid, []) if f"e_{e}" in node_to_idx]
            for eid in eids_in_chunk:
                e_idx = node_to_idx[f"e_{eid}"]
                W[c_idx, e_idx] = 1.0
                W[e_idx, c_idx] = 1.0
                edge_count += 2
                
        # 2. Document <-> Entity edges (weight = count / total_count)
        for did in all_dids:
            d_idx = node_to_idx[f"d_{did}"]
            ent_counts = doc_entity_counts.get(did, {})
            total_count = doc_total_counts.get(did, 0)
            if total_count > 0:
                for eid, count in ent_counts.items():
                    e_idx = node_to_idx[f"e_{eid}"]
                    weight = float(count) / total_count
                    W[d_idx, e_idx] = weight
                    W[e_idx, d_idx] = weight
                    edge_count += 2
                    
        row_sums = W.sum(axis=1)
        row_sums[row_sums == 0] = 1.0
        W_norm = W / row_sums[:, np.newaxis]
        
        damping = getattr(self.settings, "DAMPING", 0.5)
        
        reset_prob = np.zeros(num_nodes, dtype=np.float32)
        
        # Reset prob on Documents
        for did, score in document_dpr_scores.items():
            reset_prob[node_to_idx[f"d_{did}"]] = float(score)
            
        # Reset prob on Entities
        for eid, (score, _) in actived_entities.items():
            if f"e_{eid}" in node_to_idx:
                reset_prob[node_to_idx[f"e_{eid}"]] = float(score)
                
        sum_reset = np.sum(reset_prob)
        if sum_reset > 0:
            reset_prob = reset_prob / sum_reset
        else:
            reset_prob.fill(1.0 / num_nodes)
            
        max_pagerank_iters = getattr(self.settings, "MAX_PAGERANK_ITERS", 50)
        pr = np.copy(reset_prob)
        for _ in range(max_pagerank_iters):
            next_pr = (1.0 - damping) * reset_prob + damping * W_norm.T.dot(pr)
            if np.allclose(pr, next_pr, atol=1e-6):
                pr = next_pr
                break
            pr = next_pr
            
        final_scores = []
        for did in all_dids:
            d_idx = node_to_idx[f"d_{did}"]
            final_scores.append((did, float(pr[d_idx])))
            
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores[:top_k], num_nodes, edge_count

    async def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        t0 = time.monotonic()
        timings = {}
        
        query_embeddings = self.embedding_model.create_embeddings([query])
        query_embedding = query_embeddings[0]
        timings["embed_query"] = round((time.monotonic() - t0) * 1000)
        
        t_dpr = time.monotonic()
        dpr_hits = await self.repo.vector_search(query_embedding, top_k=50)
        
        # Aggregate chunk DPR scores into Document DPR scores
        doc_dpr_scores = {}
        for h in dpr_hits:
            did = h["document_id"]
            if did is not None:
                doc_dpr_scores[did] = max(doc_dpr_scores.get(did, 0.0), h["score"])
                
        timings["dpr"] = round((time.monotonic() - t_dpr) * 1000)

        t_seed = time.monotonic()
        seed_entity_ids, _, _, seed_entity_scores = await self.get_seed_entities(query, query_embedding)
        timings["seed_entities"] = round((time.monotonic() - t_seed) * 1000)
        
        t_iter = time.monotonic()
        actived_entities, used_chunk_ids = await self._expand_active_entities(query_embedding, seed_entity_ids, seed_entity_scores)
        timings["graph_iteration"] = round((time.monotonic() - t_iter) * 1000)
        
        t_passage = time.monotonic()
        # Fetch documents for all chunks found during graph expansion
        expanded_chunk_data = await self.repo.get_chunks_by_ids(list(used_chunk_ids))
        for cid, cdata in expanded_chunk_data.items():
            did = cdata.get("document_id")
            if did is not None and did not in doc_dpr_scores:
                doc_dpr_scores[did] = 0.0
                
        if doc_dpr_scores:
            min_dpr = min(doc_dpr_scores.values())
            max_dpr = max(doc_dpr_scores.values())
            if max_dpr > min_dpr:
                for k in doc_dpr_scores:
                    doc_dpr_scores[k] = (doc_dpr_scores[k] - min_dpr) / (max_dpr - min_dpr)
            else:
                for k in doc_dpr_scores:
                    doc_dpr_scores[k] = 1.0
                    
        all_dids = list(doc_dpr_scores.keys())
        doc_chunks_map = await self.repo.get_chunks_by_document_ids(all_dids)
        
        passage_ratio = getattr(self.settings, "PASSAGE_RATIO", 1.5)
        passage_node_weight = getattr(self.settings, "PASSAGE_NODE_WEIGHT", 0.05)
        
        for did in all_dids:
            chunks = doc_chunks_map.get(did, [])
            cids_in_doc = [c["chunk_id"] for c in chunks]
            mentions_in_doc = await self.repo.get_mentions_for_chunks(cids_in_doc)
            
            total_entity_bonus = 0.0
            
            for cid, eids in mentions_in_doc.items():
                for eid in eids:
                    if eid in actived_entities:
                        entity_score, tier = actived_entities[eid]
                        denom = tier if tier >= 1 else 1
                        entity_bonus = entity_score * math.log(1 + 1) / denom
                        total_entity_bonus += entity_bonus
                        
            dpr_score = doc_dpr_scores.get(did, 0.0)
            passage_score = passage_ratio * dpr_score + math.log(1 + total_entity_bonus)
            doc_dpr_scores[did] = passage_score * passage_node_weight
            
        timings["passage_scoring"] = round((time.monotonic() - t_passage) * 1000)
        
        t_ppr = time.monotonic()
        final_scores, vcount, ecount = await self._run_personalized_pagerank(doc_dpr_scores, used_chunk_ids, actived_entities, top_k)
        timings["ppr"] = round((time.monotonic() - t_ppr) * 1000)
        
        top_dids = [did for did, score in final_scores]
        doc_chunks_map_final = await self.repo.get_chunks_by_document_ids(top_dids)
        
        contexts = []
        for did, score in final_scores:
            chunks = doc_chunks_map_final.get(did, [])
            if chunks:
                doc_text = " ".join([c["text"] for c in chunks])
                contexts.append({
                    "chunk_id": did,
                    "document_id": did,
                    "text": doc_text,
                    "score": score
                })
                
        timings["total"] = round((time.monotonic() - t0) * 1000)
        
        return RetrievalResult(
            query=query,
            retrieval_mode="linear_rag",
            contexts=contexts,
            debug={
                "timings": timings,
                "active_entities": len(actived_entities),
                "graph_nodes": vcount,
                "graph_edges": ecount
            }
        )

