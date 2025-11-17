import os
import sys
from dotenv import load_dotenv
load_dotenv()

from utils import get_env
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception:
    from langchain.embeddings import HuggingFaceEmbeddings


from langchain_community.vectorstores import Chroma

try:
    from langchain_community.llms import Ollama
except Exception:
    try:
        from langchain_community.llms import Ollama as Ollama
    except Exception:
        try:
            from langchain_community.llms import Ollama as Ollama
        except Exception:
            raise ImportError("Ollama LangChain integration not found. Install langchain_community or langchain-ollama.")

def get_vectorstore(collection_name: str, persist_directory: str, hf_model: str):
    emb = HuggingFaceEmbeddings(model_name=hf_model, model_kwargs={"device": "cpu"})
    db = Chroma(persist_directory=persist_directory, collection_name=collection_name, embedding_function=emb)
    return db

def build_retriever_chain(llm, vectorstore, k: int = 4):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    print("debug: ",retriever)

    template = """You are given the following extracted passages from a speech by Dr. B.R. Ambedkar (the "context").
Use ONLY the context below to answer the question. If the answer is not in the context, say "I don't know based on the speech.".

Context:
{context}

Question: {question}

Answer concisely and cite the chunk indices when helpful.
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    print("DEBUG retriever =", retriever)
    print("DEBUG retriever type =", type(retriever))
    print("DEBUG retriever has invoke:", hasattr(retriever, "invoke"))

    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain
    )

    return rag_chain

def make_ollama_llm():
    base_url = get_env("OLLAMA_BASE_URL", "http://localhost:11434")
    model = get_env("OLLAMA_MODEL", "mistral")
    llm = Ollama(model=model, base_url=base_url)
    return llm

def interactive_loop(qa_chain):
    print("Interactive Q&A. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            question = input("\nYour question: ").strip()
         
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if question.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        if not question:
            print("Please type a non-empty question.")
            continue
        try:
            print("DEBUG qa_chain =", qa_chain)
            print("DEBUG type =", type(qa_chain))

            result = qa_chain.invoke({"input": question})

            print(result)
            answer = result["answer"]
            sources = result["context"]
            print("\n=== Answer ===")
            print(answer.strip())
            if sources:
                print("\n--- Retrieved chunks (short) ---")
                for i, doc in enumerate(sources):
                    snippet = doc.page_content.strip().replace("\n", " ")[:300]
                    metadata = getattr(doc, "metadata", {}) or doc.metadata
                    print(f"[{i}] chunk_id={metadata.get('chunk')} | {snippet}...")
            print("\n")
        except Exception as e:
            print("Error during query:", e)

def main():
    persist_dir = get_env("CHROMA_PERSIST_DIR", "./chroma_db")
    collection = get_env("CHROMA_COLLECTION_NAME", "ambedkar_speech")
    hf_model = get_env("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        print(f"Chroma DB appears empty at '{persist_dir}'. Run 'python src/build_vectorstore.py' first to create embeddings.")
        sys.exit(1)

    print("Loading local vector store and LLM... (this might take a moment)")
    db = get_vectorstore(collection_name=collection, persist_directory=persist_dir, hf_model=hf_model)
    print("TEST: Checking vectorstore count...")
    try:
        print("Count:", db._collection.count())
    except Exception as e:
        print("Error reading count:", e)

    llm = make_ollama_llm()
    qa_chain = build_retriever_chain(llm=llm, vectorstore=db, k=4)
    interactive_loop(qa_chain)

if __name__ == "__main__":
    main()
