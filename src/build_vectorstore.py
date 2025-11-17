import os
from dotenv import load_dotenv
load_dotenv()

from utils import read_speech_file, get_env
from ingest import split_text_into_docs

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb import PersistentClient
from chromadb.config import Settings
import chromadb
import logging
os.environ["CHROMA_LOG_LEVEL"] = "DEBUG"
logging.basicConfig(level=logging.DEBUG)


def build_vectorstore(
    speech_path="speech.txt",
    persist_directory="./chroma_db",
    collection_name="ambedkar_speech",
    hf_model="sentence-transformers/all-MiniLM-L6-v2"
):
    print("⚙️ Loading speech...")
    text = read_speech_file(speech_path)
    docs = split_text_into_docs(text)
    print(docs)
    print("Total chunks:", len(docs))

    for i, d in enumerate(docs):
        print(i, len(d["page_content"]))

    print("⚙️ Loading embeddings...")
    emb = HuggingFaceEmbeddings(model_name=hf_model)

    print("⚙️ Connecting to Chroma persistent client...")
    client = PersistentClient(path=persist_directory)

    print("⚙️ Creating/Getting vectorstore...")

    db = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=emb
    )

    print("Embedding fn:", db._embedding_function)
    print("TEST: trying to embed one text manually...")

    test_vec = emb.embed_documents(["hello world"])
    print("Got embedding:", len(test_vec[0]))


    print("📌 Adding texts...")
    print("db: ",db )
    try:
        db.add_texts(
            [d["page_content"] for d in docs],
            metadatas=[d["metadata"] for d in docs]
        )
    except Exception as e:
        print("Error adding texts:", e)
        raise e
    print(f"🎉 SUCCESS! Created vectorstore with {len(docs)} docs.")
    return db


if __name__ == "__main__":
    build_vectorstore()
