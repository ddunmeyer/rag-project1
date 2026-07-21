# rag_pipeline.py
# ---------------
# This is the heart of the RAG application.
# It orchestrates all the other modules to answer user questions.
#
# RAG stands for Retrieval-Augmented Generation:
#   1. RETRIEVAL:   Find relevant documents from our knowledge base
#   2. AUGMENTED:   Add those documents as context to our prompt
#   3. GENERATION:  Use an LLM to generate an answer based on the context
#
# This file is the central hub that grows each week:
#   Week 10: Core RAG pipeline — already complete, run it!
#   Week 11: Add conversation context    → integrate conversation.py
#   Week 12: Add input security          → integrate security.py
#   Week 13: Add hallucination monitoring → integrate monitoring.py
#   Week 14: Add filtering & fallbacks   → integrate filters.py
#   Week 15: Add query rewriting         → integrate workflow.py

from google import genai
from google.genai import types
import re

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    TOP_K_RESULTS,
    TEMPERATURE,
    SIMILARITY_THRESHOLD,
)
from embeddings import embed_text, embed_documents
from vector_store import add_documents, query_similar
from data_loader import get_documents, generate_ids
from conversation import ConversationHistory
from security import validate_input, sanitize_input
from monitoring import check_hallucination, calculate_confidence
from filters import filter_by_threshold, has_relevant_results, get_fallback_response, handle_api_error
from workflow import rewrite_query

_client = genai.Client(api_key=GEMINI_API_KEY)

# Pronouns and vague references common in follow-up questions (e.g. "what does it do?")
_VAGUE_REFERENCE = re.compile(
    r"\b(it|its|itself|they|them|their|that|this|those|these)\b",
    re.IGNORECASE,
)


def _last_user_message(conversation_history):
    """Return the most recent user message from conversation history."""
    for msg in reversed(conversation_history.messages):
        if msg["role"] == "user":
            return msg["content"]
    return None


def _build_search_query(query, conversation_history):
    """
    Build a search query for vector retrieval.

    Follow-ups like "what does it do?" embed poorly on their own. When the
    question uses a vague reference, prepend the previous user question so
    retrieval stays on the same topic (e.g. Python).
    """
    if conversation_history is None or len(conversation_history) == 0:
        return query

    if _VAGUE_REFERENCE.search(query):
        last_user = _last_user_message(conversation_history)
        if last_user:
            return f"{last_user} {query}"

    history_text = conversation_history.get_formatted_history()
    return f"{history_text}\nCurrent question: {query}"


def _resolve_question(query, conversation_history):
    """
    Clarify what the current question refers to for the LLM prompt.

    Example: "what does it do?" -> "what does it do? (about the topic from: what is python?)"
    """
    if conversation_history is None or len(conversation_history) == 0:
        return query

    if _VAGUE_REFERENCE.search(query):
        last_user = _last_user_message(conversation_history)
        if last_user:
            return f"{query} (about the topic from the previous question: {last_user})"

    return query


# ============================================================
# WEEK 10: Core RAG — Already complete. Run the app and
# explore how these three functions work together.
# ============================================================

def initialize_vector_store():
    """
    Load all sample documents, embed them, and store them in ChromaDB.
    Called once when the app starts. After this, the vector store is ready.
    """
    documents = get_documents()
    ids = generate_ids(documents)
    embeddings = embed_documents(documents)
    add_documents(documents, embeddings, ids)
    return len(documents)


def retrieve_context(query, n_results=TOP_K_RESULTS):
    """
    Find the most relevant documents for a query using semantic search.

    How it works:
      1. The query is converted to a vector embedding
      2. ChromaDB finds the document vectors closest to the query vector
      3. "Closest" means most semantically similar — not just keyword matching

    Returns:
        (documents, distances) — matched docs and their similarity distances.
        Lower distance = more similar to the query.
    """
    query_embedding = embed_text(query)
    results = query_similar(query_embedding, n_results)
    documents = results["documents"][0]
    distances = results["distances"][0]
    return documents, distances


def generate_answer(query, context_docs, conversation_history=None):
    """
    Generate an answer using Gemini with retrieved documents as context.

    The prompt includes the retrieved documents so Gemini's answer is
    grounded in our knowledge base rather than just its training data.
    """
    context = "\n\n".join(
        [f"Document {i+1}: {doc}" for i, doc in enumerate(context_docs)]
    )

    # ── Week 11 TODO ──────────────────────────────────────────────────────────
    # Add conversation history to the prompt.
    #
    # The RAG concept: LLMs have no memory between API calls. To support
    # follow-up questions, we paste the prior conversation directly into the
    # prompt so the model can see what was already discussed.
    #
    # If conversation_history is not None and len(conversation_history) > 0:
    #   history_text = conversation_history.get_formatted_history()
    #   Set history_section to: f"\nPrevious conversation:\n{history_text}\n"
    # Otherwise set history_section = ""
    #
    # Then include {history_section} in the prompt string below (already shown).
    # ─────────────────────────────────────────────────────────────────────────
    if conversation_history is not None and len(conversation_history) > 0:
        history_text = conversation_history.get_formatted_history()
        history_section = f"\nPrevious conversation:\n{history_text}\n"
    else:
        history_section = ""

    resolved_query = _resolve_question(query, conversation_history)

    prompt = f"""You are a helpful assistant that answers questions based on the provided context documents.

Context Documents:
{context}{history_section}
Current Question: {resolved_query}

Instructions:
- If the current question uses a pronoun like "it", use the previous conversation to determine what it refers to
- Answer based primarily on the provided context documents about that topic
- If the context doesn't fully answer the question, say so clearly
- Keep your answer concise and focused
- Do not make up information that isn't in the context"""

    response = _client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=TEMPERATURE),
    )
    return response.text


# ============================================================
# MAIN PIPELINE — run_rag()
# Each week you'll add one new block to this function.
# The Week 10 core at the bottom already works.
# ============================================================

def run_rag(query, conversation_history=None):
    """
    Run the full RAG pipeline for a user query.

    Returns a dictionary with:
      - "answer":     The generated answer string
      - "sources":    The source documents used
      - "distances":  Similarity distances for each source
      - "confidence": A 0–1 confidence score
      - "grounding":  Hallucination check result
      - "error":      Error message (empty string if no error)
    """

    # ── Week 12 TODO ──────────────────────────────────────────────────────────
    # Add input security before any processing happens.
    #
    # The RAG concept: always validate at the system boundary — the moment
    # user input enters the app, before it touches the LLM or vector store.
    # Prompt injection can hijack LLM behavior, so we stop bad input here.
    #
    # Steps:
    #   1. Call validate_input(query) → returns (is_valid, error_message)
    #   2. If not is_valid, return this dict immediately:
    #        {"answer": error_message, "sources": [], "distances": [],
    #         "confidence": 0.0, "grounding": {}, "error": error_message}
    #   3. Clean up the query: query = sanitize_input(query)
    # ─────────────────────────────────────────────────────────────────────────
    is_valid, error_message = validate_input(query)
    if not is_valid:
        return {
            "answer": error_message,
            "sources": [],
            "distances": [],
            "confidence": 0.0,
            "grounding": {},
            "error": error_message,
        }
    query = sanitize_input(query)

    # ── Week 15 TODO ──────────────────────────────────────────────────────────
    # Rewrite the query before retrieval to improve embedding quality.
    #
    # The RAG concept: the phrasing of the query directly affects what
    # embedding gets produced, which affects what documents get retrieved.
    # A more specific, well-formed query produces a better embedding.
    #
    # Steps:
    #   1. Get conversation context (if any):
    #        history_context = ""
    #        if conversation_history and len(conversation_history) > 0:
    #            history_context = conversation_history.get_formatted_history()
    #   2. Rewrite: query = rewrite_query(query, history_context)
    # ─────────────────────────────────────────────────────────────────────────
    history_context = ""
    if conversation_history and len(conversation_history) > 0:
        history_context = conversation_history.get_formatted_history()
    retrieval_query = rewrite_query(query, history_context)

    # ── Week 10: Core Retrieval — already complete ───────────────────────────
    # Week 11: resolve vague follow-ups (e.g. "what does it do?") using prior turns
    search_query = _build_search_query(retrieval_query, conversation_history)
    documents, distances = retrieve_context(search_query)

    # ── Week 14 TODO ──────────────────────────────────────────────────────────
    # Filter out documents that aren't similar enough to be useful.
    #
    # The RAG concept: ChromaDB always returns results even when nothing is
    # relevant. Without filtering, we might generate an answer from completely
    # unrelated documents. The threshold cuts off low-quality matches.
    #
    # Steps:
    #   1. Filter: documents, distances = filter_by_threshold(documents, distances, SIMILARITY_THRESHOLD)
    #   2. If not has_relevant_results(documents), return a fallback dict:
    #        {"answer": get_fallback_response(), "sources": [], "distances": [],
    #         "confidence": 0.0,
    #         "grounding": {"verdict": "N/A", "is_grounded": True, "warning": ""},
    #         "error": ""}
    # ─────────────────────────────────────────────────────────────────────────
    documents, distances = filter_by_threshold(documents, distances, SIMILARITY_THRESHOLD)
    if not has_relevant_results(documents):
        return {
            "answer": get_fallback_response(),
            "sources": [],
            "distances": [],
            "confidence": 0.0,
            "grounding": {"verdict": "N/A", "is_grounded": True, "warning": ""},
            "error": "",
        }

    # ── Week 10: Core Generation — already complete ──────────────────────────
    # Week 14: wrap this in try/except and call handle_api_error(e) on failure
    try:
        answer = generate_answer(query, documents, conversation_history)
    except Exception as e:
        error_msg = handle_api_error(e)
        return {
            "answer": error_msg,
            "sources": [],
            "distances": [],
            "confidence": 0.0,
            "grounding": {},
            "error": error_msg,
        }

    # ── Week 13 TODO ──────────────────────────────────────────────────────────
    # Monitor the response quality after generation.
    #
    # The RAG concept: even with context, LLMs can hallucinate. We use
    # "LLM-as-judge" — asking Gemini to evaluate its own output against the
    # source documents. We also convert vector distances into a confidence
    # score so users know how well the retrieved docs matched the query.
    #
    # Steps:
    #   1. confidence = calculate_confidence(distances)
    #   2. grounding  = check_hallucination(answer, documents)
    #   Then replace the placeholder values below with these variables.
    # ─────────────────────────────────────────────────────────────────────────
    confidence = calculate_confidence(distances)
    grounding = check_hallucination(answer, documents)

    # ── Week 11 TODO ──────────────────────────────────────────────────────────
    # Save this exchange to conversation history so follow-up questions work.
    #
    # The RAG concept: we store both sides of the exchange (user question AND
    # assistant answer) so get_formatted_history() can include both in the
    # next prompt. Without this step, history is never actually saved.
    #
    # Steps (only if conversation_history is not None):
    #   conversation_history.add_message("user", query)
    #   conversation_history.add_message("assistant", answer)
    # ─────────────────────────────────────────────────────────────────────────
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
    """
    Auto-detect which weekly features are implemented.

    Each check calls the student's code with a test value and sees
    whether it returns the placeholder or a real result. Used by the
    sidebar in app.py to show a live progress panel.
    """
    from conversation import ConversationHistory
    from security import BLOCKED_PATTERNS
    from monitoring import calculate_confidence
    from filters import filter_by_threshold
    from workflow import rewrite_query
    import inspect

    # Week 11: does get_formatted_history() produce real output?
    _h = ConversationHistory()
    _h.messages = [{"role": "user", "content": "test"}]
    week11 = _h.get_formatted_history() != ""

    # Week 12: are any injection patterns defined?
    week12 = len(BLOCKED_PATTERNS) > 0

    # Week 13: does calculate_confidence() return a non-zero value?
    week13 = calculate_confidence([0.5]) != 0.0

    # Week 14: does filter_by_threshold() actually remove high-distance docs?
    _filtered, _ = filter_by_threshold(["a", "b"], [0.3, 1.5], threshold=1.0)
    week14 = len(_filtered) == 1

    # Week 15: is rewrite_query implemented (not still a placeholder)?
    week15 = "placeholder" not in inspect.getsource(rewrite_query)

    return {
        "Week 11 — Conversation context": week11,
        "Week 12 — Input security": week12,
        "Week 13 — Hallucination monitoring": week13,
        "Week 14 — Filtering & fallbacks": week14,
        "Week 15 — Query rewriting": week15,
    }
