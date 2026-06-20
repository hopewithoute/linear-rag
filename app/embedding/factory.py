from app.embedding.contracts import EmbeddingContract
from app.embedding.openai_compatible import OpenAICompatibleEmbedding
from app.embedding.local_embedding import LocalEmbedding
from app.settings import Settings, get_settings


_embedder_cache = None

def create_embedding_runtime(settings: Settings | None = None) -> EmbeddingContract:
    global _embedder_cache
    if _embedder_cache is None:
        resolved = settings or get_settings()
        if resolved.EMBEDDING_BASE_URL:
            _embedder_cache = OpenAICompatibleEmbedding(
                base_url=resolved.EMBEDDING_BASE_URL,
                token=resolved.EMBEDDING_API_KEY,
                model=resolved.EMBEDDING_MODEL,
            )
        else:
            _embedder_cache = LocalEmbedding(model_name=resolved.EMBEDDING_MODEL)
    return _embedder_cache
