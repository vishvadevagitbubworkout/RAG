import math


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """
    Calculate the cosine similarity between two vectors.
    """

    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimensions")

    if not vector_a or not vector_b:
        raise ValueError("Vectors cannot be empty")

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Vectors cannot have zero magnitude")

    return dot_product / (magnitude_a * magnitude_b)