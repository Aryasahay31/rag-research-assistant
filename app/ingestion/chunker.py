"""
Text Chunker — splits documents into retrieval-friendly chunks.

Uses RecursiveCharacterTextSplitter which tries to split on:
  1. Double newlines (paragraph breaks) first
  2. Single newlines
  3. Spaces
  4. Characters (last resort)

This hierarchy keeps paragraphs and sentences intact when possible.

CHUNK_SIZE=800 and CHUNK_OVERLAP=100 is our sweet spot:
- 800 chars is ~150-200 tokens — small enough that the retriever
  can pinpoint relevant sections, large enough to hold context.
- 100 char overlap ensures we don't lose meaning at boundaries.
  If a key sentence sits right at a split point, the overlap
  means it appears in both chunks.
"""

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from app.config import settings


class TextChunker:
    """Split documents into overlapping chunks for embedding."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or settings.CHUNK_SIZE,
            chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
            # Split hierarchy: paragraphs → lines → sentences → words
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of Documents into smaller chunks.
        Metadata from the parent document is preserved in each chunk,
        plus we add a chunk_index so we know ordering.
        """
        all_chunks = []

        for doc in documents:
            chunks = self.splitter.split_text(doc.page_content)

            for i, chunk_text in enumerate(chunks):
                # Inherit parent metadata + add chunk-specific info
                metadata = {**doc.metadata, "chunk_index": i}
                all_chunks.append(Document(
                    page_content=chunk_text,
                    metadata=metadata,
                ))

        return all_chunks