import argparse
import sys
from pathlib import Path

import ollama

from app.embedding.embedder import EMBEDDING_MODEL
from app.rag import GENERATION_MODEL, RAGPipeline


def main() -> int:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Ask questions about text files using your local Ollama models."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "txt",
        help="Folder containing .txt files (default: ./txt)",
    )
    args = parser.parse_args()

    try:
        ollama.show(EMBEDDING_MODEL)
        ollama.show(GENERATION_MODEL)
    except ollama.ResponseError as error:
        print(f"Ollama model error: {error}")
        print(
            "Check the model names in OLLAMA_EMBEDDING_MODEL and "
            "OLLAMA_GENERATION_MODEL, or pull the missing model with "
            "`ollama pull <model>`."
        )
        return 1

    pipeline = RAGPipeline(chunk_size=500, overlap=80, top_k=4)
    try:
        document_count = pipeline.load_directory(args.data_dir)
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        print(f"Could not load text files: {error}", file=sys.stderr)
        return 1
    print(f"Loaded {document_count} text file(s) from {args.data_dir.resolve()}.")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Answer model: {GENERATION_MODEL}")
    print("Ask a question, or type 'exit' to quit.")

    while True:
        try:
            query = input("\nQuestion: ").strip()
        except EOFError:
            print()
            return 0

        if query.lower() in {"exit", "quit"}:
            return 0
        if not query:
            continue

        print("\nAnswer: ", end="", flush=True)
        pipeline.query(
            query,
            on_token=lambda token: print(token, end="", flush=True),
        )
        print()


if __name__ == "__main__":
    raise SystemExit(main())
