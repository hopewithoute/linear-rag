from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    FRUGAL_DATABASE_URL: Optional[str] = None
    FRUGAL_API_KEY: Optional[str] = None
    DB_SCHEMA: str = "linear_rag"

    GLINER_MODEL: str = "urchade/gliner_medium-v2.1"
    NER_BACKEND: str = "spacy"  # spacy or gliner
    NER_LABELS: str = ""        # comma-separated labels
    TERM_EXTRACTOR_BASE_URL: Optional[str] = None
    TERM_EXTRACTOR_API_KEY: Optional[str] = None
    TERM_EXTRACTOR_THRESHOLD: float = 0.5
    SPACY_MODEL: str = "en_core_web_sm"

    EMBEDDING_BASE_URL: Optional[str] = None
    EMBEDDING_API_KEY: Optional[str] = None
    EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-0.6B"
    EMBEDDING_DIMENSIONS: int = 1024

    # Generator (LLM answer generation)
    GENERATOR_BASE_URL: Optional[str] = None
    GENERATOR_API_KEY: Optional[str] = None
    GENERATOR_MODEL: str = "deepseek-v4-flash"

    # RAG Chunking
    CHUNK_SIZE_TOKENS: int = 200
    CHUNK_OVERLAP_TOKENS: int = 50

    PPR_ALPHA: float = 0.15
    PPR_MAX_ITER: int = 3
    PPR_MAX_FRONTIER_NODES: int = 50000
    PPR_MAX_FRONTIER_EDGES: int = 250000
    PPR_MAX_HOPS: int = 2

    MAX_ENTITIES_PER_CHUNK: int = 32
    MIN_ENTITY_LENGTH: int = 2
    MAX_ENTITY_DEGREE_PER_QUERY: int = 500

    SNAPSHOT_DIR: str = "./snapshots"
    SNAPSHOT_COMPILE_MODE: str = "full"

    VECTOR_TOP_K: int = 50
    BM25_TOP_K: int = 50
    FINAL_TOP_K: int = 12
    RRF_K: int = 60
    DENSE_WEIGHT: float = 0.6
    BM25_WEIGHT: float = 0.4

    TOMBSTONE_COMPACTION_THRESHOLD: float = 0.30

    # Reranker
    RERANKER_MODE: str = "heuristic"  # none, heuristic, cross_encoder
    RERANKER_CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_BOOST_EXACT_MATCH: float = 0.1
    RERANKER_BOOST_SHORT_CHUNK: float = 0.02
    RERANKER_PENALIZE_LONG_CHUNK: float = 0.02

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
