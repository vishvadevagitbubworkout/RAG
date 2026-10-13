from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VectorStore:
    """Simple in-memory vector store for a RAG pipeline."""

    documents: list[str] = field(default_factory=list)
    indexed_documents: list[dict] = field(default_factory=list)

    def add_document(self, document: str) -> None:
        if not document or not document.strip():
            return
        self.documents.append(document)

    def add_documents(self, documents: list[str]) -> None:
        for document in documents:
            self.add_document(document)

    def set_documents(self, documents: list[str]) -> None:
        self.documents = [document for document in documents if document and document.strip()]
        self.indexed_documents = []

    def clear(self) -> None:
        self.documents.clear()
        self.indexed_documents.clear()

    def as_texts(self) -> list[str]:
        return list(self.documents)
