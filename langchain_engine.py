# langchain_engine.py
# -------------------
# LangChain components used by the production RAG pipeline.
#
# Replaces direct google-genai + manual Chroma calls in the main app path.
# Weekly modules (security, filters, monitoring, workflow) still wrap this engine.

from __future__ import annotations

import re
import time

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.prebuilt import create_react_agent

from config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_MAX_RETRIES,
    TEMPERATURE,
    TOP_K_RESULTS,
)
from data_loader import get_documents

_vectorstore: Chroma | None = None
_react_agent = None


def _parse_retry_seconds(error: Exception) -> float:
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(error), re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1
    return 15


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_llm(temperature: float = TEMPERATURE) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=temperature,
    )


def get_vectorstore() -> Chroma:
    """Return the shared persisted Chroma vector store."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_PERSIST_DIR,
        )
    return _vectorstore


def initialize_vector_store() -> int:
    """Load documents into Chroma on first run; reuse persisted vectors after."""
    vectorstore = get_vectorstore()
    count = vectorstore._collection.count()
    if count > 0:
        return count

    documents = [Document(page_content=text) for text in get_documents()]
    vectorstore.add_documents(documents)
    return len(documents)


def retrieve_with_scores(query: str, k: int = TOP_K_RESULTS) -> tuple[list[str], list[float]]:
    """Semantic search with L2 distance scores for filtering and confidence."""
    results = get_vectorstore().similarity_search_with_score(query, k=k)
    documents = [doc.page_content for doc, _ in results]
    distances = [float(distance) for _, distance in results]
    return documents, distances


def invoke_llm_text(prompt: str, *, temperature: float = TEMPERATURE) -> str:
    """Call Gemini through LangChain with rate-limit retries."""
    llm = get_llm(temperature)
    last_error = None

    for attempt in range(GEMINI_MAX_RETRIES):
        try:
            response = llm.invoke(prompt)
            return _message_text(response.content)
        except Exception as error:
            last_error = error
            error_text = str(error).lower()
            is_rate_limit = any(
                token in error_text
                for token in ("429", "resource_exhausted", "rate limit", "quota")
            )
            if is_rate_limit and attempt < GEMINI_MAX_RETRIES - 1:
                time.sleep(_parse_retry_seconds(error))
                continue
            raise

    raise last_error  # pragma: no cover


def _message_text(content) -> str:
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


def generate_answer(
    query: str,
    context_docs: list[str],
    conversation_history=None,
    *,
    resolve_question,
) -> str:
    """Generate a grounded answer with retrieved context and optional chat history."""
    context = "\n\n".join(f"Document {i + 1}: {doc}" for i, doc in enumerate(context_docs))

    history_section = ""
    if conversation_history is not None and len(conversation_history) > 0:
        history_section = f"\nPrevious conversation:\n{conversation_history.get_formatted_history()}\n"

    resolved_query = resolve_question(query, conversation_history)

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant that answers questions based on the provided context documents.

Context Documents:
{context}{history_section}
Current Question: {question}

Instructions:
- If the current question uses a pronoun like "it", use the previous conversation to determine what it refers to
- Answer based primarily on the provided context documents about that topic
- If the context doesn't fully answer the question, say so clearly
- Keep your answer concise and focused
- Do not make up information that isn't in the context"""
    )

    chain = prompt | get_llm() | StrOutputParser()
    return chain.invoke(
        {
            "context": context,
            "history_section": history_section,
            "question": resolved_query,
        }
    )


def get_react_agent():
    """ReAct agent that decides when to search the knowledge base."""
    global _react_agent
    if _react_agent is not None:
        return _react_agent

    vectorstore = get_vectorstore()

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the tech docs knowledge base for facts about Python, ML, RAG, and AI."""
        docs, _ = retrieve_with_scores(query)
        if not docs:
            return "No relevant documents found."
        return "\n\n".join(docs)

    _react_agent = create_react_agent(
        get_llm(),
        tools=[search_knowledge_base],
        prompt=(
            "You are a helpful assistant for a RAG learning project. "
            "Use search_knowledge_base before answering factual questions. "
            "If the knowledge base does not contain the answer, say so clearly. "
            "Do not invent facts."
        ),
    )
    return _react_agent


def run_react_answer(query: str, conversation_history=None) -> str:
    """Run the ReAct agent and return the final assistant message text."""
    agent = get_react_agent()
    user_content = query
    if conversation_history is not None and len(conversation_history) > 0:
        user_content = (
            f"Previous conversation:\n{conversation_history.get_formatted_history()}\n\n"
            f"Current question: {query}"
        )

    result = agent.invoke({"messages": [("user", user_content)]})
    final = result["messages"][-1]
    if isinstance(final, AIMessage):
        return _message_text(final.content)
    return _message_text(getattr(final, "content", final))
