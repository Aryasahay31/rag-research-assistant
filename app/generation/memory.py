"""
Conversation Memory — tracks chat history per session.

Stores the last N messages so users can ask follow-up questions:

User: "What does the document say about caching?"
Bot:  "The document describes Redis-based caching for..."
User: "How is the TTL configured?"  <-- follow-up, needs context

Without memory, the LLM wouldn't know "the TTL" refers to the
caching discussion. With memory, it sees the full conversation.

We keep only the last MAX_HISTORY messages to avoid:
1. Exceeding the LLM's context window
2. Confusing the LLM with old, irrelevant context
3. Wasting tokens (and money) on old messages
"""

from typing import List, Dict
from collections import defaultdict


class ConversationMemory:
    """In-memory conversation history per session."""

    MAX_HISTORY = 10  # keep last 10 messages (5 Q&A pairs)

    def __init__(self):
        # Dict of session_id -> list of messages
        self._history: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to a session's history."""
        self._history[session_id].append({
            "role": role,
            "content": content,
        })

        # Trim to keep only the last MAX_HISTORY messages
        if len(self._history[session_id]) > self.MAX_HISTORY:
            self._history[session_id] = self._history[session_id][-self.MAX_HISTORY:]

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a session."""
        return self._history.get(session_id, [])

    def clear(self, session_id: str):
        """Clear history for a session."""
        self._history.pop(session_id, None)

    def clear_all(self):
        """Clear all session histories."""
        self._history.clear()