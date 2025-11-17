from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict

def split_text_into_docs(text: str, chunk_size: int = 300, chunk_overlap: int = 50):
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_text(text)

    docs = [
        {"page_content": c, "metadata": {"chunk": i}}
        for i, c in enumerate(chunks)
    ]

    return docs