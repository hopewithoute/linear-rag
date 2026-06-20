import httpx
from typing import List, Optional, Union
from app.embedding.contracts import EmbeddingContract


class OpenAICompatibleEmbedding(EmbeddingContract):
    def __init__(self, base_url: Optional[str] = None, token: Optional[str] = None, model: Optional[str] = None):
        if not base_url:
            raise RuntimeError("EMBEDDING_BASE_URL is not configured")
        url = base_url.rstrip("/")
        self.url = f"{url}/embeddings" if not url.endswith("/embeddings") else url
        self.token = token or "dev-secret-key"
        self.model = model or "local-embedding-model"
        self._client = httpx.Client(timeout=300.0)

    def create_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {"input": texts, "model": self.model}

        try:
            response = self._client.post(self.url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return [item.get("embedding", []) for item in data.get("data", [])]
        except httpx.TimeoutException:
            raise RuntimeError("Embedding service timed out (300s)")
        except Exception as exc:
            raise RuntimeError(f"HTTP embedding inference failed: {exc}") from exc
