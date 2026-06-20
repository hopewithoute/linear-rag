from app.chunking.contracts import ChunkerContract
from app.chunking.sentence_chunker import SentenceChunker
from app.settings import Settings, get_settings


def create_chunker(settings: Settings | None = None) -> ChunkerContract:
    return SentenceChunker()
