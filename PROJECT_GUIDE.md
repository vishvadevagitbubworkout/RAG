# Local RAG Project: Architecture and Code Guide

This guide explains how the project answers questions from local text files,
how each algorithm works, and what each Python file is responsible for. It is
intended as a study and presentation guide.

## 1. What the project does

This is a local Retrieval-Augmented Generation (RAG) application. It combines:

- **Retrieval:** Find passages in your own documents that are relevant to a
  question.
- **Augmented generation:** Give those passages and the question to a local
  language model, which writes an answer grounded in that context.

The app uses Ollama for both the embedding model and the answer-generation
model. The interactive entry point is `run_rag.py`. It reads `.txt` files
under `txt/`, including nested folders, and streams the generated answer to
the terminal.

## 2. Architecture at a glance

```text
txt/**/*.txt
     |
     v
load_text_file() --> RAGPipeline.load_directory()
     |
     v
word-based chunk_text() with overlap
     |
     v
index_chunks() --> Ollama embedding model
     |
     v
in-memory VectorStore

User question
     |
     v
embed_text(question) --> Ollama embedding model
     |
     v
retrieve_top_k() --> cosine similarity against indexed chunks
     |
     v
join best passages into context
     |
     v
Ollama generation model (stream=True)
     |
     +--> on_token callback --> print each arriving piece
     |
     v
complete answer and query result
```

There is no database or persistent vector index: documents and embeddings
remain in memory for one app run and are rebuilt at startup.

## 3. The complete data flow

### Startup and ingestion

1. `run_rag.py` selects the input folder (by default, the project's `txt/`).
2. It checks that Ollama can access the configured embedding and generation
   models.
3. `RAGPipeline.load_directory()` finds `.txt` files recursively and reads
   non-empty contents as UTF-8.
4. `chunk_text()` splits each document into word-based chunks, keeping some
   words from the previous chunk as overlap.
5. `index_chunks()` sends each chunk to Ollama's embedding endpoint.
6. `VectorStore` keeps the original document strings and the indexed
   `{text, embedding}` chunk records in memory.

### Query and streamed answer

1. The app reads a question from the terminal.
2. The same embedding model converts the question into a vector.
3. `retrieve_top_k()` computes cosine similarity between the question vector
   and every stored chunk vector, sorts by score, and keeps the best `top_k`.
4. The selected text chunks are joined into the context supplied to the
   generator, along with the question.
5. The app calls Ollama generation with `stream=True`.
6. For each response piece, the RAG pipeline appends it to the full answer and
   calls `on_token`. The CLI callback prints the piece immediately.
7. When streaming ends, `query()` returns its regular dictionary containing
   the question, retrieved results, context, and complete answer.

The callback is optional. Existing code that calls `pipeline.query(question)`
still gets a completed result and uses a non-streaming Ollama response.

## 4. Algorithms and key ideas

### 4.1 Word-based chunking

`chunk_text(text, chunk_size, overlap)` first splits the document on
whitespace. If `chunk_size` is \(C\) words and `overlap` is \(O\) words, each
new chunk starts \(C-O\) words after the previous start:

```text
step = chunk_size - overlap
chunk[i] = words[i * step : i * step + chunk_size]
```

The overlap keeps nearby context from being lost at a chunk boundary. The
function rejects non-positive chunk sizes, negative overlap, and overlap that
is greater than or equal to the chunk size. A final chunk may be shorter than
the configured maximum.

### 4.2 Embeddings

An embedding is a numeric vector representing text in a space where texts with
similar meanings are expected to have vectors pointing in similar directions.
`embed_text()` calls the configured Ollama embedding model for both document
chunks and user questions. Using the same model for both is important because
their vectors must inhabit the same vector space and have matching dimensions.

### 4.3 Cosine similarity

For vectors \(a\) and \(b\), the project calculates:

```text
cosine_similarity(a, b) = dot(a, b) / (||a|| * ||b||)
```

The result is a direction-based similarity score: 1 means same direction, 0
means perpendicular, and -1 means opposite direction. The implementation
rejects vectors with different dimensions, empty vectors, and zero-magnitude
vectors.

### 4.4 Top-k retrieval

For each indexed chunk, the retriever calculates its cosine score against the
question embedding. It sorts all scored chunks from highest to lowest and
returns at most `top_k`.

If there are \(N\) chunks, each with \(d\) dimensions, scoring is approximately
\(O(Nd)\) and sorting is \(O(N \log N)\). This is an exact, brute-force search,
not an approximate-nearest-neighbor index.

### 4.5 Retrieval-augmented generation

The generator receives the user question and the text of the retrieved
chunks. Its prompt instructs it to answer in English using that context and
to say when the answer is not present. Retrieval supplies evidence; generation
turns that evidence into a readable answer. The model can still make mistakes,
so answers should be checked against the displayed source material when
accuracy matters.

### 4.6 Streaming

Ollama streaming produces a sequence of partial generation responses rather
than one complete response. The RAG pipeline processes each part in order:

```text
partial answer -> append to answer buffer -> call on_token(partial answer)
```

The interactive CLI's callback prints each part with `flush=True`, so the
terminal updates as generation proceeds. The pipeline still collects all parts
so callers receive the same complete `answer` field when the query finishes.

## 5. Configuration and running the app

The defaults are:

- Embeddings: `nomic-embed-text:latest`
- Answer generation: `rafw007/qwen35-claude-coder:4b`
- CLI chunk size: 500 words
- CLI overlap: 80 words
- CLI retrieval count: 4

Set `OLLAMA_EMBEDDING_MODEL` and `OLLAMA_GENERATION_MODEL` before launching
Python to select other installed models.

```powershell
python -m pip install -r requirements.txt
ollama serve
python run_rag.py
```

Put documents in `txt/`. To use a different folder:

```powershell
python run_rag.py --data-dir .\txt\my-subfolder
```

Type a question and then `exit` or `quit` to leave the interactive app.

## 6. Python file-by-file guide

### Application and command-line files

| File | Responsibility |
|------|----------------|
| `run_rag.py` | Command-line app. Parses the input-folder option, verifies the Ollama models, loads/indexes `.txt` files, reads questions, and prints answer tokens as they arrive. |
| `app/rag.py` | Main orchestration layer. `RAGPipeline` loads documents, chunks/indexes them, embeds questions, retrieves context, calls Ollama, optionally streams tokens through a callback, and returns query results. Also defines `build_rag_pipeline()` and `query_rag()` convenience helpers. |
| `app/vector_store.py` | In-memory data container for original documents and indexed chunk records. Provides add, set, clear, and text-access operations; it does not persist data or perform similarity search itself. |
| `app/__init__.py` | Marks `app` as a Python package. It intentionally exports no extra API. |

### Ingestion files

| File | Responsibility |
|------|----------------|
| `app/ingestion/loader.py` | Reads one UTF-8 text file and returns its contents. File errors are allowed to surface to the caller. |
| `app/ingestion/chunker.py` | Implements overlapping, word-based chunking and validates the chunk and overlap settings. |

### Embedding files

| File | Responsibility |
|------|----------------|
| `app/embedding/embedder.py` | Reads the embedding model name from `OLLAMA_EMBEDDING_MODEL` (with a local default), validates non-empty input, and calls Ollama to produce a vector. |
| `app/embedding/indexer.py` | Skips blank chunks and creates `{text, embedding}` records by embedding each non-empty chunk. |
| `app/embedding/__init__.py` | Marks `app.embedding` as a Python package; no symbols are re-exported. |

### Retrieval files

| File | Responsibility |
|------|----------------|
| `app/retrieval/similarity.py` | Implements cosine similarity and checks vector dimensions and magnitude. |
| `app/retrieval/retriever.py` | Scores indexed chunks against a query vector, sorts by score, and returns the requested top-k results. |
| `app/retrieval/__init__.py` | Marks `app.retrieval` as a Python package; no symbols are re-exported. |

`requirements.txt` lists the Python package required by the app. The Ollama
application and the configured embedding/generation models are installed
separately.

## 7. Data structures and result format

Each indexed chunk is a dictionary:

```python
{
    "text": "The original chunk text",
    "embedding": [0.12, -0.04, ...],
}
```

`retrieve_top_k()` adds a `score` to each returned record. `RAGPipeline.query()`
returns:

```python
{
    "query": "The user's question",
    "results": [{"text": "...", "embedding": [...], "score": 0.8}],
    "context": "Text assembled from the best chunks",
    "answer": "The complete generated answer",
}
```

## 8. Current limitations and next steps

- The in-memory index is rebuilt at every launch; a persistent vector database
  would avoid re-embedding documents every time.
- Retrieval compares the query with every chunk, which becomes slower as the
  collection grows.
- Chunks are split by word count, not by sentence, section, or document layout.
- Retrieved chunks do not currently include source filenames in the answer
  metadata, so the CLI cannot cite which file supplied each passage.
- The CLI displays the answer, but not the retrieved passages. The Python
  result does include both `context` and scored `results`.
- A language model can produce unsupported statements even when prompted to
  use retrieved context; evaluation and source citations would improve
  trustworthiness.
