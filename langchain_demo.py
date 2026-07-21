# langchain_demo.py
# -----------------
# Week 15.5 (Optional): Rebuild the basic RAG flow using LangChain.
#
# This does NOT replace app.py or rag_pipeline.py — it is a side-by-side demo
# so you can compare manual implementation vs framework abstractions.
#
# Run:
#   pip install -r requirements-langchain.txt
#   python langchain_demo.py

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    TEMPERATURE,
    TOP_K_RESULTS,
)
from data_loader import get_documents


def format_docs(documents):
    """Join retrieved documents into one context string."""
    return "\n\n".join(doc.page_content for doc in documents)


def build_langchain_rag_chain():
    """
    Build the same core flow as rag_pipeline.py using LangChain components.

    Manual project          →  LangChain equivalent
    ----------------          ---------------------
    data_loader.py          →  Document objects
    embeddings.py           →  HuggingFaceEmbeddings
    vector_store.py         →  Chroma vector store + retriever
    generate_answer() prompt →  ChatPromptTemplate
    Gemini API call         →  ChatGoogleGenerativeAI
    run_rag() orchestration →  LCEL chain (retriever | prompt | llm | parser)
    """
    documents = [Document(page_content=text) for text in get_documents()]

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=f"langchain_{COLLECTION_NAME}",
        persist_directory="./chroma_db_langchain",
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant that answers questions based on the provided context documents.

Context Documents:
{context}

Question: {question}

Instructions:
- Answer based primarily on the provided context documents
- If the context doesn't fully answer the question, say so clearly
- Keep your answer concise and focused
- Do not make up information that isn't in the context"""
    )

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=TEMPERATURE,
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def main():
    print("Building LangChain RAG chain (first run embeds documents)...")
    chain, retriever = build_langchain_rag_chain()

    questions = [
        "What is Python?",
        "What else can it do in the real world?",
        "Who won the Super Bowl this year?",
    ]

    print("\n=== LangChain RAG Demo ===\n")

    for question in questions:
        print(f"Question: {question}")

        retrieved = retriever.invoke(question)
        print(f"Retrieved {len(retrieved)} document(s)")
        if retrieved:
            print(f"  Top source: {retrieved[0].page_content[:90]}...")

        answer = chain.invoke(question)
        print(f"Answer: {answer}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
