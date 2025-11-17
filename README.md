"""ABOUT"""
# Ambedkar Q&A (Local RAG) — Prototype

This repo builds a local Retrieval-Augmented Generation (RAG) pipeline that:
- ingests a short speech (speech.txt),
- splits it into chunks,
- creates embeddings (sentence-transformers/all-MiniLM-L6-v2),
- stores them in a local Chroma vector DB,
- uses Ollama (Mistral 7B) as the LLM to answer questions using retrieved context.

## Requirements
- Python 3.8+
- Ollama installed & running locally (see instructions below). Ollama binds to http://localhost:11434 by default.
- Enough disk to store model weights you pull with Ollama.

Ask questions interactively. Type `exit` or `quit` to stop.

## Files
- `src/build_vectorstore.py`: load speech.txt -> split -> embed -> persist.
- `src/qa_cli.py`: interactive CLI: retrieves relevant chunks and queries Ollama via LangChain.
- `src/ingest.py`, `src/utils.py`: helper utilities.

## Troubleshooting
- If embedding fails, ensure `torch` and `sentence-transformers` installed.
- If Ollama connection fails, make sure `ollama serve` is running and `OLLAMA_BASE_URL` matches.

## Citations
- Ollama runs on `http://localhost:11434` by default. (Ollama docs)
- Embeddings use `sentence-transformers/all-MiniLM-L6-v2`.
- Chroma persistence via LangChain.

## Caution
Download version compatible libraries based on your pc requirements otherwise Chroma silently crashes.

Enjoy!

"""SETUP INSTRUCTIONS"""

# Create & activate venv, install:

python -m venv .venv
venv/bin/activate
pip install -r requirements.txt


# Set up Ollama & pull mistral:

ollama pull mistral
ollama serve

# Build vectors:

python src/build_vectorstore.py

# Run Q&A:

python src/qa_cli.py
