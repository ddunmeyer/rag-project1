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
from compliance import (
    log_compliance_event,
    metadata_summary,
    redact_for_display,
    tag_model_output,
    tag_retrieved_chunk,
    tag_user_input,
)
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


def _error_result(message: str, *, compliance: dict | None = None) -> dict:
    return {
        "answer": message,
        "sources": [],
        "distances": [],
        "confidence": 0.0,
        "grounding": {},
        "error": message,
        "compliance": compliance or {},
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
    documents, distances, stored_metadatas = retrieve_with_scores(search_query)
    filtered_docs, filtered_distances = filter_by_threshold(
        documents, distances, SIMILARITY_THRESHOLD
    )
    retrieval_tags = []
    source_metadatas = []
    rank = 0
    for doc, distance, stored in zip(documents, distances, stored_metadatas):
        if distance <= SIMILARITY_THRESHOLD:
            rank += 1
            tag = tag_retrieved_chunk(doc, rank=rank)
            if stored:
                tag.tags.extend(
                    f"stored:{key}={value}"
                    for key, value in stored.items()
                    if key != "compliance_version"
                )
            retrieval_tags.append(tag)
            source_metadatas.append(tag.to_dict())

    return filtered_docs, filtered_distances, retrieval_tags, source_metadatas


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
        log_compliance_event("input_blocked", query)
        return _error_result(error_message)

    query = sanitize_input(query)
    input_metadata = tag_user_input(query)
    log_compliance_event("user_input_received", query, input_metadata)

    search_query = _prepare_retrieval_query(query, conversation_history)

    try:
        if RAG_MODE == "react":
            answer = run_react_answer(query, conversation_history)
            documents, distances, retrieval_tags, source_metadatas = _retrieve_filtered(
                search_query
            )
        else:
            documents, distances, retrieval_tags, source_metadatas = _retrieve_filtered(
                search_query
            )
            if not has_relevant_results(documents):
                log_compliance_event("retrieval_fallback", search_query, input_metadata)
                return {
                    "answer": get_fallback_response(),
                    "sources": [],
                    "distances": [],
                    "confidence": 0.0,
                    "grounding": {"verdict": "N/A", "is_grounded": True, "warning": ""},
                    "error": "",
                    "compliance": {
                        "input": input_metadata.to_dict(),
                        "retrieved": metadata_summary([]),
                    },
                }
            log_compliance_event(
                "documents_retrieved",
                f"{len(documents)} chunks",
                tag_retrieved_chunk(documents[0], rank=1) if documents else None,
            )
            answer = generate_answer(
                query,
                documents,
                conversation_history,
                resolve_question=_resolve_question,
            )
    except Exception as error:
        safe_error = redact_for_display(str(error))
        log_compliance_event("pipeline_error", safe_error, input_metadata)
        return _error_result(handle_api_error(error), compliance={"input": input_metadata.to_dict()})

    output_metadata = tag_model_output(answer)
    log_compliance_event("model_output", answer, output_metadata)

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
        "compliance": {
            "input": input_metadata.to_dict(),
            "retrieved": metadata_summary(retrieval_tags),
            "output": output_metadata.to_dict(),
            "source_metadatas": source_metadatas,
        },
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

    from compliance import classify_text, redact_for_log, DataSource, Sensitivity

    sample_meta = classify_text("hello", DataSource.USER_INPUT)
    week18 = (
        sample_meta.sensitivity == Sensitivity.PUBLIC
        and "[REDACTED]" in redact_for_log("contact test@example.com")
    )

    from pathlib import Path

    week19 = Path("tests/test_basic.py").exists() and Path(
        ".github/workflows/tests.yml"
    ).exists()

    return {
        "Week 11 — Conversation context": week11,
        "Week 12 — Input security": week12,
        "Week 13 — Hallucination monitoring": week13,
        "Week 14 — Filtering & fallbacks": week14,
        "Week 15 — Query rewriting": week15,
        "Week 15.5 — LangChain pipeline": langchain_ready,
        "Week 17 — Data protection": week17,
        "Week 18 — Compliance tagging & redaction": week18,
        "Week 19 — Tests & CI/CD": week19,
    }
