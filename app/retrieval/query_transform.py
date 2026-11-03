"""
Query Transformer — HyDE using Ollama (local LLM).
"""

import requests
from app.config import settings


class QueryTransformer:
    """Transform user queries using HyDE for better retrieval."""

    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL

    def _call_ollama(self, messages: list) -> str:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": settings.LLM_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.7},
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def hyde_transform(self, query: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable assistant. Given a question, "
                    "write a detailed paragraph that would answer it. "
                    "Write as if you are writing a textbook passage. "
                    "Do not say 'I think' or 'I believe'. Just state facts."
                ),
            },
            {"role": "user", "content": query},
        ]
        return self._call_ollama(messages)

    def multi_query(self, query: str, n_queries: int = 3) -> list:
        messages = [
            {
                "role": "system",
                "content": (
                    f"Generate {n_queries} different versions of the "
                    "given question. Each version should approach the "
                    "topic from a different angle to help find relevant "
                    "documents. Return ONLY the questions, one per line."
                ),
            },
            {"role": "user", "content": query},
        ]
        result = self._call_ollama(messages)
        queries = result.strip().split("\n")
        queries = [q.lstrip("0123456789.-) ").strip() for q in queries if q.strip()]
        return queries[:n_queries]