# config.py
# ---------
# Central configuration file for the RAG project.
# Keeping all settings in one place makes it easy to change them
# without hunting through multiple files.

import os
from dotenv import load_dotenv

# Load environment variables from a .env file in the project root.
# This is how we keep secrets (like API keys) out of our code.
load_dotenv()

# --- API Keys ---
# Your Gemini API key, loaded from the .env file.
# Never put your actual key directly in code — it could get shared by accident.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Model Settings ---
# The name of the AI model we use for generating answers.
# gemini-3.5-flash-lite is fast, cost-efficient, and works well on the free tier.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# The name of the embedding model we use to turn text into numbers.
# "all-MiniLM-L6-v2" is small, fast, and works well for semantic search.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Vector Store Settings ---
# The name of our ChromaDB collection (like a table in a database).
COLLECTION_NAME = "tech_docs"

# How many documents to retrieve from the vector store per query.
# Retrieving top 3 gives the LLM enough context without overwhelming it.
TOP_K_RESULTS = 3

# The similarity threshold for filtering results.
# ChromaDB returns L2 distances: 0 = identical, 2 = completely different.
# We keep results where the distance is below this number.
# 1.0 is a reasonable default — anything further is probably not relevant.
SIMILARITY_THRESHOLD = 1.0

# --- Generation Settings ---
# Temperature controls how "creative" the AI is when generating answers.
# 0.2 is low — answers will be more factual and consistent.
TEMPERATURE = 0.2

# --- Conversation Settings ---
# How many recent messages to include in the conversation history.
# Keeping only the last 10 turns prevents the context from growing too large.
MAX_HISTORY_TURNS = 10

# --- API Usage Settings ---
# The free Gemini tier allows very few requests per minute/day.
# Each question can use up to 3 API calls (rewrite + answer + hallucination check).
# These settings reduce calls so the app stays usable on the free tier.
def _env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# Only rewrite vague follow-up questions (saves 1 API call on clear first questions).
ENABLE_QUERY_REWRITING = _env_bool("ENABLE_QUERY_REWRITING", True)

# LLM-as-judge adds 1 extra API call per answer. Off by default on free tier.
ENABLE_HALLUCINATION_CHECK = _env_bool("ENABLE_HALLUCINATION_CHECK", False)

# Retry Gemini calls when rate-limited instead of failing immediately.
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
