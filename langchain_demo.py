# langchain_demo.py
# -----------------
# Week 15.5 (Optional): Rebuild the basic RAG flow using LangChain.
#
# This does NOT replace app.py or rag_pipeline.py — it is a side-by-side demo
# so you can compare manual implementation vs framework abstractions.
#
# Run:
#   pip install -r requirements-langchain.txt
#   python langchain_demo.py              # both demos
#   python langchain_demo.py --mode chain  # LCEL RAG chain only
#   python langchain_demo.py --mode react  # ReAct agent only

import argparse

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.prebuilt import create_react_agent

from config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    TEMPERATURE,
    TOP_K_RESULTS,
)
from data_loader import get_documents

DEMO_QUESTIONS = [
    "What is Python?",
    "What else can it do in the real world?",
    "Who won the Super Bowl this year?",
]


def format_docs(documents):
    """Join retrieved documents into one context string."""
    return "\n\n".join(doc.page_content for doc in documents)


def message_text(content):
    """Normalize Gemini/LangChain message content to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def build_retriever():
    """Create the shared Chroma retriever used by both demos."""
    documents = [Document(page_content=text) for text in get_documents()]

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=f"langchain_{COLLECTION_NAME}",
        persist_directory="./chroma_db_langchain",
    )

    return vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})


def build_llm():
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=TEMPERATURE,
    )


def build_langchain_rag_chain(retriever):
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

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | build_llm()
        | StrOutputParser()
    )


def build_react_agent(retriever):
    """
    Build a ReAct agent that decides when to call the retriever tool.

    ReAct loop:
      Thought → Action (tool call) → Observation (tool result) → repeat until done

    Uses LangGraph's create_react_agent. LangChain 1.x also exposes the newer
    create_agent API, but this helper is the classic ReAct pattern.
    """

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the project's tech docs knowledge base for relevant information.

        Use this when you need facts about Python, RAG, embeddings, vector databases,
        LangChain, or other topics covered in the course materials.
        """
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant documents found."
        return format_docs(docs)

    return create_react_agent(
        build_llm(),
        tools=[search_knowledge_base],
        prompt=(
            "You are a helpful assistant for a RAG learning project. "
            "Use the search_knowledge_base tool to look up facts before answering. "
            "If the knowledge base does not contain the answer, say so clearly. "
            "Do not invent facts."
        ),
    )


def print_react_trace(messages):
    """Print the agent's tool-use steps in a readable ReAct-style trace."""
    step = 0
    for message in messages:
        if isinstance(message, HumanMessage):
            continue

        if isinstance(message, AIMessage):
            text = message_text(message.content).strip()
            if text:
                step += 1
                print(f"  Thought {step}: {text}")

            for tool_call in message.tool_calls or []:
                print(
                    f"  Action {step}: {tool_call['name']}("
                    f"{tool_call['args']})"
                )

        elif isinstance(message, ToolMessage):
            preview = message_text(message.content).replace("\n", " ")[:160]
            print(f"  Observation: {preview}...")

    final = messages[-1]
    if isinstance(final, AIMessage):
        print(f"  Final answer: {message_text(final.content)}")


def run_chain_demo(chain, retriever, questions):
    print("\n=== LangChain LCEL RAG Demo ===\n")

    for question in questions:
        print(f"Question: {question}")

        retrieved = retriever.invoke(question)
        print(f"Retrieved {len(retrieved)} document(s)")
        if retrieved:
            print(f"  Top source: {retrieved[0].page_content[:90]}...")

        answer = chain.invoke(question)
        print(f"Answer: {answer}\n")
        print("-" * 60)


def run_react_demo(agent, questions):
    print("\n=== LangChain ReAct Agent Demo ===\n")

    for question in questions:
        print(f"Question: {question}")
        result = agent.invoke({"messages": [("user", question)]})
        print_react_trace(result["messages"])
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="LangChain RAG and ReAct demos")
    parser.add_argument(
        "--mode",
        choices=("chain", "react", "both"),
        default="both",
        help="Run the LCEL chain demo, the ReAct agent demo, or both",
    )
    args = parser.parse_args()

    print("Building LangChain components (first run embeds documents)...")
    retriever = build_retriever()

    if args.mode in ("chain", "both"):
        chain = build_langchain_rag_chain(retriever)
        run_chain_demo(chain, retriever, DEMO_QUESTIONS)

    if args.mode in ("react", "both"):
        agent = build_react_agent(retriever)
        run_react_demo(agent, DEMO_QUESTIONS)


if __name__ == "__main__":
    main()
