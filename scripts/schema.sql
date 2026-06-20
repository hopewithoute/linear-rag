-- LinearRag-RAG Schema
-- Run: psql -f scripts/schema.sql <database_url>

CREATE SCHEMA IF NOT EXISTS linear_rag;

CREATE TABLE IF NOT EXISTS linear_rag.documents (
    id BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT '',
    source_uri TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS linear_rag.chunks (
    chunk_id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES linear_rag.documents(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    start_offset INT NOT NULL,
    end_offset INT NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    embedding HALFVEC(1024),
    search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS linear_rag_chunks_document_idx ON linear_rag.chunks(document_id);
CREATE INDEX IF NOT EXISTS linear_rag_chunks_active_idx ON linear_rag.chunks(active);
CREATE INDEX IF NOT EXISTS linear_rag_chunks_search_idx ON linear_rag.chunks USING GIN(search_vector);

-- Run after pgvector extension is enabled:
-- CREATE INDEX IF NOT EXISTS linear_rag_chunks_embedding_idx ON linear_rag.chunks
--   USING HNSW (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE TABLE IF NOT EXISTS linear_rag.entities (
    entity_id BIGSERIAL PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_type TEXT,
    document_frequency INT NOT NULL DEFAULT 0,
    degree INT NOT NULL DEFAULT 0,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS linear_rag_entities_name_idx ON linear_rag.entities(canonical_name);

CREATE TABLE IF NOT EXISTS linear_rag.entity_aliases (
    alias TEXT PRIMARY KEY,
    entity_id BIGINT NOT NULL REFERENCES linear_rag.entities(entity_id) ON DELETE CASCADE,
    confidence FLOAT NOT NULL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS linear_rag.mentions (
    mention_id BIGSERIAL PRIMARY KEY,
    chunk_id BIGINT NOT NULL REFERENCES linear_rag.chunks(chunk_id) ON DELETE CASCADE,
    entity_id BIGINT NOT NULL REFERENCES linear_rag.entities(entity_id) ON DELETE CASCADE,
    surface_text TEXT,
    confidence FLOAT NOT NULL DEFAULT 1.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS linear_rag_mentions_chunk_idx ON linear_rag.mentions(chunk_id);
CREATE INDEX IF NOT EXISTS linear_rag_mentions_entity_idx ON linear_rag.mentions(entity_id);

CREATE TABLE IF NOT EXISTS linear_rag.graph_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    version TEXT NOT NULL,
    num_chunk_nodes BIGINT NOT NULL,
    num_entity_nodes BIGINT NOT NULL,
    num_directed_edges BIGINT NOT NULL,
    format TEXT NOT NULL DEFAULT 'csr',
    storage_uri TEXT,
    active BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS linear_rag.retrieval_audit (
    id BIGSERIAL PRIMARY KEY,
    tenant_id TEXT,
    query TEXT NOT NULL,
    chunk_ids BIGINT[],
    scores FLOAT[],
    latency_ms INT,
    snapshot_id BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
