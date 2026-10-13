from app.embedding.embedder import embed_text


def index_chunks(chunks: list[str]) -> list[dict]:
    """
    Generate embeddings for text chunks.

    Returns:
        A list of dictionaries containing text and embeddings.
    """

    indexed_chunks = []

    for chunk in chunks:
        if not chunk.strip():
            continue

        indexed_chunks.append(
            {
                "text": chunk,
                "embedding": embed_text(chunk),
            }
        )

    return indexed_chunks