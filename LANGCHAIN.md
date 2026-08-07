# LangChain in This Project

LangChain is the **production pipeline** for this app — not a separate demo.

## Architecture

| Component | LangChain class |
|-----------|-----------------|
| `langchain_engine.py` | Core engine: embeddings, Chroma, LLM, LCEL chain, ReAct agent |
| `rag_pipeline.py` | Orchestration + weekly modules (security, filters, monitoring, workflow) |
| `workflow.py` | Query rewrite/decomposition via `invoke_llm_text()` |
| `monitoring.py` | Hallucination judge via `invoke_llm_text()` |

Legacy reference files (`embeddings.py`, `vector_store.py`, `gemini_utils.py`) remain for learning but are no longer used by the main app path.

## Run the app

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Pipeline modes

Set in `.env`:

```env
RAG_MODE=chain   # default — retrieve, filter, LCEL answer generation
RAG_MODE=react   # LangGraph ReAct agent with search_knowledge_base tool
```

## CLI test

```powershell
python langchain_demo.py "What is Python?"
```

Uses the same `run_rag()` as the Streamlit app.

## Concept mapping (historical)

| Manual code (Weeks 10–15) | Now handled by |
|---------------------------|----------------|
| `embeddings.py` | `HuggingFaceEmbeddings` in `langchain_engine.py` |
| `vector_store.py` | `Chroma` + `similarity_search_with_score` |
| `generate_answer()` | `ChatPromptTemplate` + LCEL chain |
| Gemini raw API | `ChatGoogleGenerativeAI` |
| `run_rag()` orchestration | `rag_pipeline.py` + LangChain engine |

Security, filtering, confidence, and query rewriting still run in Python modules **around** the LangChain engine — the same pattern used in production RAG systems.
