import os

import ollama


EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")


def embed_text(text: str) -> list[float]:
    """Create an embedding with the configured local Ollama model."""
    if not text.strip():
        raise ValueError("Text cannot be empty")

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return list(response.embeddings[0])
