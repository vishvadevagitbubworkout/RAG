
def chunk_text(
    text: str,
    chunk_size: int = 100,
    overlap: int = 20,
) -> list[str]:
    """
    Split text into overlapping word-based chunks.

    Args:
        text: The input document text.
        chunk_size: Maximum number of words per chunk.
        overlap: Number of overlapping words between chunks.

    Returns:
        A list of text chunks.

    Raises:
        ValueError: If chunk_size or overlap is invalid.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        # Stop after reaching the end of the document.
        if end >= len(words):
            break

        start += chunk_size - overlap

    return chunks