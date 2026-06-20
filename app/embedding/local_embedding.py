from typing import List, Union
from sentence_transformers import SentenceTransformer
from app.embedding.contracts import EmbeddingContract
import logging

logger = logging.getLogger(__name__)

class LocalEmbedding(EmbeddingContract):
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        logger.info(f"Loading local SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def create_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
