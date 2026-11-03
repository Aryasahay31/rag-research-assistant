"""
Document Loader — reads raw text from uploaded files.

Supports PDF, DOCX, and TXT. Each loader returns a list of Document
objects (LangChain's standard format) with the text and metadata
like source filename and page number.

WHY NOT use LangChain's built-in loaders?
We could, but writing our own is ~30 lines and gives us full control
over error handling and metadata — important for interviews.
"""

import os
from typing import List
from langchain.schema import Document
from PyPDF2 import PdfReader
from docx import Document as DocxDocument


class DocumentLoader:
    """Load documents from various file formats."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    def load(self, file_path: str) -> List[Document]:
        """
        Load a single file and return a list of Documents.
        Each page/section becomes one Document with metadata.
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return self._load_pdf(file_path)
        elif ext == ".docx":
            return self._load_docx(file_path)
        elif ext == ".txt":
            return self._load_txt(file_path)
        else:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

    def load_directory(self, dir_path: str) -> List[Document]:
        """Load all supported files from a directory."""
        all_docs = []
        for filename in os.listdir(dir_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in self.SUPPORTED_EXTENSIONS:
                file_path = os.path.join(dir_path, filename)
                all_docs.extend(self.load(file_path))
        return all_docs

    def _load_pdf(self, file_path: str) -> List[Document]:
        """
        Extract text page by page from a PDF.
        Each page becomes a separate Document so we can track
        which page a chunk originally came from.
        """
        reader = PdfReader(file_path)
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():  # skip blank pages
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(file_path),
                        "page": i + 1,
                        "total_pages": len(reader.pages),
                    }
                ))
        return docs

    def _load_docx(self, file_path: str) -> List[Document]:
        """Extract text from a Word document."""
        doc = DocxDocument(file_path)
        # Combine all paragraphs into one document
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        if text:
            return [Document(
                page_content=text,
                metadata={"source": os.path.basename(file_path)}
            )]
        return []

    def _load_txt(self, file_path: str) -> List[Document]:
        """Read a plain text file."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        if text.strip():
            return [Document(
                page_content=text,
                metadata={"source": os.path.basename(file_path)}
            )]
        return []