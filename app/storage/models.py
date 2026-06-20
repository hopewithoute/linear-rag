import uuid
from datetime import datetime
from sqlalchemy import (
    Column, BigInteger, Integer, String, Text, DateTime, Float, Boolean,
    ForeignKey, Index, func, Computed,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, ARRAY
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import HALFVEC
from app.settings import get_settings

settings = get_settings()

Base = declarative_base()


class LinearRagBase(Base):
    __abstract__ = True
    __table_args__ = {"schema": settings.DB_SCHEMA}


class Document(LinearRagBase):
    __tablename__ = "documents"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id = Column(String, nullable=False, server_default="")
    source_uri = Column(Text, nullable=True)
    metadata_ = Column("metadata", JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("linear_rag_documents_tenant_idx", "tenant_id"),
        {"schema": settings.DB_SCHEMA},
    )


class Chunk(LinearRagBase):
    __tablename__ = "chunks"

    chunk_id = Column(BigInteger, primary_key=True, autoincrement=True)
    document_id = Column(BigInteger, ForeignKey(f"{settings.DB_SCHEMA}.documents.id", ondelete="CASCADE"), nullable=False)
    text = Column(Text, nullable=False)
    start_offset = Column(Integer, nullable=False)
    end_offset = Column(Integer, nullable=False)
    active = Column(Boolean, nullable=False, server_default="true")
    embedding = Column(HALFVEC(settings.EMBEDDING_DIMENSIONS), nullable=True)
    search_vector = Column(TSVECTOR, Computed("to_tsvector('simple', text)", persisted=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index("linear_rag_chunks_document_idx", "document_id"),
        Index("linear_rag_chunks_active_idx", "active"),
        Index("linear_rag_chunks_embedding_idx", "embedding", postgresql_using="hnsw",
              postgresql_with={"m": 16, "ef_construction": 64},
              postgresql_ops={"embedding": "halfvec_cosine_ops"}),
        Index("linear_rag_chunks_search_idx", "search_vector", postgresql_using="gin"),
        {"schema": settings.DB_SCHEMA},
    )


class Entity(LinearRagBase):
    __tablename__ = "entities"

    entity_id = Column(BigInteger, primary_key=True, autoincrement=True)
    canonical_name = Column(Text, nullable=False)
    entity_type = Column(Text, nullable=True)
    document_frequency = Column(Integer, nullable=False, server_default="0")
    degree = Column(Integer, nullable=False, server_default="0")
    active = Column(Boolean, nullable=False, server_default="true")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("linear_rag_entities_name_idx", "canonical_name"),
        {"schema": settings.DB_SCHEMA},
    )


class EntityAlias(LinearRagBase):
    __tablename__ = "entity_aliases"

    alias = Column(Text, primary_key=True)
    entity_id = Column(BigInteger, ForeignKey(f"{settings.DB_SCHEMA}.entities.entity_id", ondelete="CASCADE"), nullable=False)
    confidence = Column(Float, nullable=False, server_default="1.0")


class Mention(LinearRagBase):
    __tablename__ = "mentions"

    mention_id = Column(BigInteger, primary_key=True, autoincrement=True)
    chunk_id = Column(BigInteger, ForeignKey(f"{settings.DB_SCHEMA}.chunks.chunk_id", ondelete="CASCADE"), nullable=False)
    entity_id = Column(BigInteger, ForeignKey(f"{settings.DB_SCHEMA}.entities.entity_id", ondelete="CASCADE"), nullable=False)
    surface_text = Column(Text, nullable=True)
    confidence = Column(Float, nullable=False, server_default="1.0")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("linear_rag_mentions_chunk_idx", "chunk_id"),
        Index("linear_rag_mentions_entity_idx", "entity_id"),
        {"schema": settings.DB_SCHEMA},
    )


class GraphSnapshot(LinearRagBase):
    __tablename__ = "graph_snapshots"

    snapshot_id = Column(BigInteger, primary_key=True, autoincrement=True)
    version = Column(Text, nullable=False)
    num_chunk_nodes = Column(BigInteger, nullable=False)
    num_entity_nodes = Column(BigInteger, nullable=False)
    num_directed_edges = Column(BigInteger, nullable=False)
    format = Column(Text, nullable=False, server_default="csr")
    storage_uri = Column(Text, nullable=True)
    active = Column(Boolean, nullable=False, server_default="false")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class RetrievalAudit(LinearRagBase):
    __tablename__ = "retrieval_audit"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id = Column(Text, nullable=True)
    query = Column(Text, nullable=False)
    chunk_ids = Column(ARRAY(BigInteger), nullable=True)
    scores = Column(ARRAY(Float), nullable=True)
    latency_ms = Column(Integer, nullable=True)
    snapshot_id = Column(BigInteger, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
