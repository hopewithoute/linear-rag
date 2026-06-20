"""LLM answer generation for LinearRag-RAG.
Generates answers from retrieved contexts using LLM API.
"""
import httpx
from typing import List, Dict, Any, Optional


class LLMGenerator:
    """Generate answers from contexts using LLM API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        from app.settings import get_settings
        settings = get_settings()
        self.base_url = base_url or settings.GENERATOR_BASE_URL or "https://api.deepseek.com/v1"
        self.api_key = api_key or settings.GENERATOR_API_KEY or ""
        self.model = model or settings.GENERATOR_MODEL or "deepseek-v4-flash"

    def generate(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        """Generate answer from query and contexts."""
        if not self.api_key:
            raise RuntimeError("GENERATOR_API_KEY not configured")

        # Build context text
        context_text = "\n\n".join(
            f"[Source {i+1}]: {ctx.get('text', '')}"
            for i, ctx in enumerate(contexts[:5])  # Limit to 5 contexts
        )

        # Build prompt
        prompt = f"""Answer the following question based on the provided context. 
If the context doesn't contain enough information, say "I don't have enough information to answer this question."

Context:
{context_text}

Question: {query}

Answer:"""

        # Call LLM API
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error generating answer: {e}"


def generate_answer(
    query: str,
    contexts: List[Dict[str, Any]],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> str:
    """Convenience function to generate answer."""
    generator = LLMGenerator(base_url=base_url, api_key=api_key, model=model)
    return generator.generate(query, contexts, max_tokens=max_tokens, temperature=temperature)
