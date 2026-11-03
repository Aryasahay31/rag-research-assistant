"""
Prompt Builder — constructs the final prompt sent to the LLM.

This is where retrieved chunks become part of the conversation.
The prompt has 3 parts:

1. SYSTEM: Instructions telling the LLM how to behave
   - Use only the provided context
   - Say "I don't know" if context doesn't answer the question
   - Cite which source document the answer came from

2. CONTEXT: The top chunks from our retrieval pipeline, formatted
   with source metadata so the LLM can cite them

3. CONVERSATION: Chat history + the current question

WHY "say I don't know" matters:
Without this instruction, the LLM will hallucinate an answer using
its training data. The whole point of RAG is GROUNDING answers in
your documents. If the docs don't cover it, we want honesty.
"""

from typing import List, Dict, Any


class PromptBuilder:
    """Build prompts for the RAG pipeline."""

    SYSTEM_PROMPT = """You are a helpful research assistant. Answer questions based ONLY on the provided context documents.

Rules:
1. If the context contains the answer, provide a clear and detailed response.
2. If the context does NOT contain enough information, say "I don't have enough information in the provided documents to answer this question."
3. Always cite which source document(s) your answer comes from using [Source: filename] format.
4. If multiple documents are relevant, synthesize information from all of them.
5. Do not make up information that is not in the context.
6. Be concise but thorough."""

    def build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into a context string.

        Each chunk gets a header with its source and page number
        so the LLM knows where to attribute information.
        """
        if not chunks:
            return "No relevant documents found."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk["metadata"].get("source", "Unknown")
            page = chunk["metadata"].get("page", "")
            page_str = f", Page {page}" if page else ""

            context_parts.append(
                f"--- Document {i} [Source: {source}{page_str}] ---\n"
                f"{chunk['text']}\n"
            )

        return "\n".join(context_parts)

    def build_messages(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        chat_history: List[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        """
        Build the full message list for the OpenAI API.

        Structure:
        [system, context_as_user_msg, ...chat_history, current_question]
        """
        context = self.build_context(chunks)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Here are the relevant documents:\n\n{context}",
            },
            {
                "role": "assistant",
                "content": "I've reviewed the documents. What would you like to know?",
            },
        ]

        # Add chat history (previous Q&A pairs)
        if chat_history:
            for msg in chat_history:
                messages.append(msg)

        # Add current question
        messages.append({"role": "user", "content": query})

        return messages