"""Initial linear_rag-rag schema

Revision ID: a1b2c3d4e5f6
Revises: 
Create Date: 2026-06-18 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from app.settings import get_settings
settings = get_settings()
schema_name = settings.DB_SCHEMA
from sqlalchemy.dialects import postgresql
import pgvector

revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('CREATE SCHEMA IF NOT EXISTS linear_rag;')
    op.execute('CREATE EXTENSION IF NOT EXISTS vector;')

    op.create_table('documents',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('tenant_id', sa.String(), server_default='', nullable=False),
        sa.Column('source_uri', sa.Text(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        schema=schema_name
    )
    op.create_index('linear_rag_documents_tenant_idx', 'documents', ['tenant_id'], unique=False, schema=schema_name)

    op.create_table('chunks',
        sa.Column('chunk_id', sa.BigInteger(), nullable=False),
        sa.Column('document_id', sa.BigInteger(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('start_offset', sa.Integer(), nullable=False),
        sa.Column('end_offset', sa.Integer(), nullable=False),
        sa.Column('active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('embedding', pgvector.sqlalchemy.halfvec.HALFVEC(dim=1024), nullable=True),
        sa.Column('search_vector', postgresql.TSVECTOR(), sa.Computed("to_tsvector('simple', text)", persisted=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], [f"{schema_name}.documents.id"], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('chunk_id'),
        schema=schema_name
    )
    op.create_index('linear_rag_chunks_document_idx', 'chunks', ['document_id'], unique=False, schema=schema_name)
    op.create_index('linear_rag_chunks_active_idx', 'chunks', ['active'], unique=False, schema=schema_name)
    op.create_index('linear_rag_chunks_embedding_idx', 'chunks', ['embedding'], unique=False, schema=schema_name,
                     postgresql_using='hnsw', postgresql_with={'m': 16, 'ef_construction': 64},
                     postgresql_ops={'embedding': 'halfvec_cosine_ops'})
    op.create_index('linear_rag_chunks_search_idx', 'chunks', ['search_vector'], unique=False, schema=schema_name,
                     postgresql_using='gin')

    op.create_table('entities',
        sa.Column('entity_id', sa.BigInteger(), nullable=False),
        sa.Column('canonical_name', sa.Text(), nullable=False),
        sa.Column('entity_type', sa.Text(), nullable=True),
        sa.Column('document_frequency', sa.Integer(), server_default='0', nullable=False),
        sa.Column('degree', sa.Integer(), server_default='0', nullable=False),
        sa.Column('active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('entity_id'),
        schema=schema_name
    )
    op.create_index('linear_rag_entities_name_idx', 'entities', ['canonical_name'], unique=False, schema=schema_name)

    op.create_table('entity_aliases',
        sa.Column('alias', sa.Text(), nullable=False),
        sa.Column('entity_id', sa.BigInteger(), nullable=False),
        sa.Column('confidence', sa.Float(), server_default='1.0', nullable=False),
        sa.ForeignKeyConstraint(['entity_id'], [f"{schema_name}.entities.entity_id"], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('alias'),
        schema=schema_name
    )

    op.create_table('mentions',
        sa.Column('mention_id', sa.BigInteger(), nullable=False),
        sa.Column('chunk_id', sa.BigInteger(), nullable=False),
        sa.Column('entity_id', sa.BigInteger(), nullable=False),
        sa.Column('surface_text', sa.Text(), nullable=True),
        sa.Column('confidence', sa.Float(), server_default='1.0', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['chunk_id'], [f"{schema_name}.chunks.chunk_id"], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['entity_id'], [f"{schema_name}.entities.entity_id"], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('mention_id'),
        schema=schema_name
    )
    op.create_index('linear_rag_mentions_chunk_idx', 'mentions', ['chunk_id'], unique=False, schema=schema_name)
    op.create_index('linear_rag_mentions_entity_idx', 'mentions', ['entity_id'], unique=False, schema=schema_name)

    op.create_table('graph_snapshots',
        sa.Column('snapshot_id', sa.BigInteger(), nullable=False),
        sa.Column('version', sa.Text(), nullable=False),
        sa.Column('num_chunk_nodes', sa.BigInteger(), nullable=False),
        sa.Column('num_entity_nodes', sa.BigInteger(), nullable=False),
        sa.Column('num_directed_edges', sa.BigInteger(), nullable=False),
        sa.Column('format', sa.Text(), server_default='csr', nullable=False),
        sa.Column('storage_uri', sa.Text(), nullable=True),
        sa.Column('active', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('snapshot_id'),
        schema=schema_name
    )

    op.create_table('retrieval_audit',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('tenant_id', sa.Text(), nullable=True),
        sa.Column('query', sa.Text(), nullable=False),
        sa.Column('chunk_ids', postgresql.ARRAY(sa.BigInteger()), nullable=True),
        sa.Column('scores', postgresql.ARRAY(sa.Float()), nullable=True),
        sa.Column('latency_ms', sa.Integer(), nullable=True),
        sa.Column('snapshot_id', sa.BigInteger(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        schema=schema_name
    )


def downgrade() -> None:
    op.drop_table('retrieval_audit', schema=schema_name)
    op.drop_table('graph_snapshots', schema=schema_name)
    op.drop_index('linear_rag_mentions_entity_idx', table_name='mentions', schema=schema_name)
    op.drop_index('linear_rag_mentions_chunk_idx', table_name='mentions', schema=schema_name)
    op.drop_table('mentions', schema=schema_name)
    op.drop_table('entity_aliases', schema=schema_name)
    op.drop_index('linear_rag_entities_name_idx', table_name='entities', schema=schema_name)
    op.drop_table('entities', schema=schema_name)
    op.drop_index('linear_rag_chunks_search_idx', table_name='chunks', schema=schema_name, postgresql_using='gin')
    op.drop_index('linear_rag_chunks_embedding_idx', table_name='chunks', schema=schema_name,
                   postgresql_using='hnsw', postgresql_with={'m': 16, 'ef_construction': 64},
                   postgresql_ops={'embedding': 'halfvec_cosine_ops'})
    op.drop_index('linear_rag_chunks_active_idx', table_name='chunks', schema=schema_name)
    op.drop_index('linear_rag_chunks_document_idx', table_name='chunks', schema=schema_name)
    op.drop_table('chunks', schema=schema_name)
    op.drop_index('linear_rag_documents_tenant_idx', table_name='documents', schema=schema_name)
    op.drop_table('documents', schema=schema_name)
    op.execute('DROP SCHEMA IF EXISTS linear_rag CASCADE;')
