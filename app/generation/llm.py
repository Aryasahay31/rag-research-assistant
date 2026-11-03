"""
RAG Chain — now using Ollama (local LLM, completely free).
Ollama runs models like Llama 3.2 on your machine.
"""

import requests
from typing import List, Dict, Any
from app.config import settings
from app.generation.prompt import PromptBuilder
from app.generation.memory import ConversationMemory


class RAGChain:
    """Generate answers from retrieved context using Ollama."""

    def __init__(self):
        self.prompt_builder = PromptBuilder()
        self.memory = ConversationMemory()
        self.base_url = settings.OLLAMA_BASE_URL

    def generate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        session_id: str = "default",
        use_memory: bool = True,
    ) -> str:
        chat_history = []
        if use_memory:
            chat_history = self.memory.get_history(session_id)

        messages = self.prompt_builder.build_messages(
            query=query,
            chunks=chunks,
            chat_history=chat_history,
        )

        # Call Ollama's chat API
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": settings.LLM_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                },
            },
            timeout=120,
        )
        response.raise_for_status()

        answer = response.json()["message"]["content"]

        if use_memory:
            self.memory.add_message(session_id, "user", query)
            self.memory.add_message(session_id, "assistant", answer)

        return answer

    def generate_streaming(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        session_id: str = "default",
    ):
        chat_history = self.memory.get_history(session_id)

        messages = self.prompt_builder.build_messages(
            query=query,
            chunks=chunks,
            chat_history=chat_history,
        )

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": settings.LLM_MODEL,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": 0.3,
                },
            },
            stream=True,
            timeout=120,
        )
        response.raise_for_status()

        import json
        full_answer = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    token = data["message"]["content"]
                    full_answer += token
                    yield token

        self.memory.add_message(session_id, "user", query)
        self.memory.add_message(session_id, "assistant", full_answer)