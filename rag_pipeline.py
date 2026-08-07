# rag_pipeline.py
# ---------------
# Orchestrates the LangChain-backed RAG pipeline and weekly feature modules.

import re

from config import (
    ENABLE_HALLUCINATION_CHECK,
    ENABLE_QUERY_REWRITING,
    ENABLE_RESTRICTED_TOPIC_CHECKS,
    ENABLE_SENSITIVE_DATA_CHECKS,
    RAG_MODE,
    SIMILARITY_THRESHOLD,
)
from filters import (
    filter_by_threshold,
    get_fallback_response,
    handle_api_error,
    has_relevant_results,
)
from langchain_engine import (
    generate_answer,
    initialize_vector_store,
    retrieve_with_scores,
    run_react_answer,
)
from monitoring import calculate_confidence, check_hallucination
from security import sanitize_input, validate_input
from workflow import rewrite_query

_VAGUE_REFERENCE = re.compile(
    r"\b(it|its|itself|they|them|their|that|this|those|these)\b",
    re.IGNORECASE,
)


def _last_user_message(conversation_history):
    for msg in reversed(conversation_history.messages):
        if msg["role"] == "user":
            return msg["content"]
    return None


def _build_search_query(query, conversation_history):
    if conversation_history is None or len(conversation_history) == 0:
        return query

    if _VAGUE_REFERENCE.search(query):
        last_user = _last_user_message(conversation_history)
        if last_user:
            return f"{last_user} {query}"

    history_text = conversation_history.get_formatted_history()
    return f"{history_text}\nCurrent question: {query}"


def _resolve_question(query, conversation_history):
    if conversation_history is None or len(conversation_history) == 0:
        return query

    if _VAGUE_REFERENCE.search(query):
        last_user = _last_user_message(conversation_history)
        if last_user:
            return f"{query} (about the topic from the previous question: {last_user})"

    return query


def _needs_query_rewrite(query, conversation_history):
    if conversation_history is not None and len(conversation_history) > 0:
        return True
    return bool(_VAGUE_REFERENCE.search(query))


def _error_result(message: str) -> dict:
    return {
        "answer": message,
        "sources": [],
        "distances": [],
        "confidence": 0.0,
        "grounding": {},
        "error": message,
    }


def _prepare_retrieval_query(query, conversation_history):
    history_context = ""
    if conversation_history and len(conversation_history) > 0:
        history_context = conversation_history.get_formatted_history()

    retrieval_query = query
    if ENABLE_QUERY_REWRITING and _needs_query_rewrite(query, conversation_history):
        retrieval_query = rewrite_query(query, history_context)

    return _build_search_query(retrieval_query, conversation_history)


def _retrieve_filtered(search_query):
    documents, distances = retrieve_with_scores(search_query)
    return filter_by_threshold(documents, distances, SIMILARITY_THRESHOLD)


def run_rag(query, conversation_history=None):
    """
    Run the full LangChain RAG pipeline for a user query.

    RAG_MODE:
      - "chain" (default): retrieve → filter → LangChain LCEL generation
      - "react": LangGraph ReAct agent with search_knowledge_base tool
    """
    is_valid, error_message = validate_input(
        query,
        check_sensitive_data=ENABLE_SENSITIVE_DATA_CHECKS,
        check_restricted_topics=ENABLE_RESTRICTED_TOPIC_CHECKS,
    )
    if not is_valid:
        return _error_result(error_message)

    query = sanitize_input(query)
    search_query = _prepare_retrieval_query(query, conversation_history)

    try:
        if RAG_MODE == "react":
            answer = run_react_answer(query, conversation_history)
            documents, distances = _retrieve_filtered(search_query)
        else:
            documents, distances = _retrieve_filtered(search_query)
            if not has_relevant_results(documents):
                return {
                    "answer": get_fallback_response(),
                    "sources": [],
                    "distances": [],
                    "confidence": 0.0,
                    "grounding": {"verdict": "N/A", "is_grounded": True, "warning": ""},
                    "error": "",
                }
            answer = generate_answer(
                query,
                documents,
                conversation_history,
                resolve_question=_resolve_question,
            )
    except Exception as error:
        return _error_result(handle_api_error(error))

    confidence = calculate_confidence(distances) if documents else 0.0
    if ENABLE_HALLUCINATION_CHECK and documents:
        grounding = check_hallucination(answer, documents)
    else:
        grounding = {"verdict": "SKIPPED", "is_grounded": True, "warning": ""}

    if conversation_history is not None:
        conversation_history.add_message("user", query)
        conversation_history.add_message("assistant", answer)

    return {
        "answer": answer,
        "sources": documents,
        "distances": distances,
        "confidence": confidence,
        "grounding": grounding,
        "error": "",
    }


def get_feature_status():
    """Auto-detect implemented weekly features for the Streamlit sidebar."""
    from conversation import ConversationHistory
    from monitoring import calculate_confidence
    from filters import filter_by_threshold
    from workflow import rewrite_query
    from security import BLOCKED_PATTERNS, validate_input
    import inspect

    _h = ConversationHistory()
    _h.messages = [{"role": "user", "content": "test"}]
    week11 = _h.get_formatted_history() != ""

    week12 = (
        len(BLOCKED_PATTERNS) > 0
        and validate_input("contact me at test@example.com", check_injection=False)[0] is False
    )

    week13 = calculate_confidence([0.5]) != 0.0

    _filtered, _ = filter_by_threshold(["a", "b"], [0.3, 1.5], threshold=1.0)
    week14 = len(_filtered) == 1

    week15 = "placeholder" not in inspect.getsource(rewrite_query)

    week17 = validate_input("contact me at test@example.com", check_injection=False)[0] is False

    try:
        from langchain_engine import get_vectorstore

        langchain_ready = get_vectorstore() is not None
    except Exception:
        langchain_ready = False

    return {
        "Week 11 — Conversation context": week11,
        "Week 12 — Input security": week12,
        "Week 13 — Hallucination monitoring": week13,
        "Week 14 — Filtering & fallbacks": week14,
        "Week 15 — Query rewriting": week15,
        "Week 15.5 — LangChain pipeline": langchain_ready,
        "Week 17 — Data protection": week17,
    }
