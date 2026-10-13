from app.retrieval.similarity import cosine_similarity


def retrieve_top_k(
    query_embedding: list[float],
    documents: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """
    Return the most similar documents for a query embedding.
    """

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    scored_documents = []

    for document in documents:
        score = cosine_similarity(
            query_embedding,
            document["embedding"],
        )

        scored_documents.append(
            {
                "text": document["text"],
                "embedding": document["embedding"],
                "score": score,
            }
        )

    scored_documents.sort(
        key=lambda document: document["score"],
        reverse=True,
    )

    return scored_documents[:top_k]