from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Callable

import ollama

from app.embedding.embedder import embed_text
from app.embedding.indexer import index_chunks
from app.ingestion.chunker import chunk_text
from app.ingestion.loader import load_text_file
from app.retrieval.retriever import retrieve_top_k
from app.vector_store import VectorStore


GENERATION_MODEL = os.getenv(
    "OLLAMA_GENERATION_MODEL",
    "rafw007/qwen35-claude-coder:4b",
)


class RAGPipeline:
    """Full retrieval-augmented generation pipeline."""

    def __init__(
        self,
        documents: list[str] | None = None,
        chunk_size: int = 100,
        overlap: int = 20,
        top_k: int = 3,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.store = VectorStore()

        if documents:
            self.store.set_documents(documents)
            self.index_documents()

    def add_document(self, document: str) -> None:
        self.store.add_document(document)
        self.index_documents()

    def add_documents(self, documents: list[str]) -> None:
        self.store.add_documents(documents)
        self.index_documents()

    def load_file(self, file_path: str | Path) -> None:
        text = load_text_file(str(file_path))
        self.add_document(text)

    def load_directory(self, directory_path: str | Path) -> int:
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Text directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        files = sorted(directory.rglob("*.txt"))
        if not files:
            raise ValueError(f"No .txt files found in {directory}")

        documents = [
            load_text_file(str(file_path)).strip()
            for file_path in files
        ]
        documents = [document for document in documents if document]
        if not documents:
            raise ValueError(f"No non-empty .txt files found in {directory}")

        self.add_documents(documents)
        return len(documents)

    def index_documents(self) -> None:
        if not self.store.documents:
            self.store.indexed_documents = []
            return

        chunks = []
        for document in self.store.documents:
            chunks.extend(
                chunk_text(
                    document,
                    chunk_size=self.chunk_size,
                    overlap=self.overlap,
                )
            )

        if not chunks:
            self.store.indexed_documents = []
            return

        self.store.indexed_documents = index_chunks(chunks)

    def query(
        self,
        query: str,
        top_k: int | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> dict:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not self.store.indexed_documents:
            raise ValueError("No indexed documents available")

        query_embedding = embed_text(query)

        results = retrieve_top_k(
            query_embedding=query_embedding,
            documents=self.store.indexed_documents,
            top_k=top_k or self.top_k,
        )

        context = "\n\n".join(item["text"] for item in results)
        answer = self.generate_answer(query, context, results, on_token=on_token)

        return {
            "query": query,
            "results": results,
            "context": context,
            "answer": answer,
        }

    def generate_answer(
        self,
        query: str,
        context: str,
        results: list[dict],
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        if not results:
            return "I could not find any relevant information in the indexed documents."

        prompt = (
            "You are a helpful assistant. Answer in English using only "
            "the context below. If the answer is not present, say so.\n\n"
            f"Question: {query}\n\nContext:\n{context}"
        )
        if on_token is None:
            response = ollama.generate(model=GENERATION_MODEL, prompt=prompt)
            answer = response.response.strip()
        else:
            answer_parts = []
            for response in ollama.generate(
                model=GENERATION_MODEL,
                prompt=prompt,
                stream=True,
            ):
                token = response.response
                if token:
                    answer_parts.append(token)
                    on_token(token)
            answer = "".join(answer_parts).strip()

        if not answer:
            raise RuntimeError(
                f"Ollama model '{GENERATION_MODEL}' returned an empty answer."
            )
        return answer


def build_rag_pipeline(
    documents: list[str] | str | None = None,
    chunk_size: int = 100,
    overlap: int = 20,
    top_k: int = 3,
) -> RAGPipeline:
    if isinstance(documents, str):
        documents = [documents]

    return RAGPipeline(
        documents=documents,
        chunk_size=chunk_size,
        overlap=overlap,
        top_k=top_k,
    )


def query_rag(
    query: str,
    documents: list[str] | str,
    chunk_size: int = 100,
    overlap: int = 20,
    top_k: int = 3,
) -> dict:
    pipeline = build_rag_pipeline(
        documents=documents,
        chunk_size=chunk_size,
        overlap=overlap,
        top_k=top_k,
    )
    return pipeline.query(query, top_k=top_k)
