# Week 15.5 — LangChain Introduction (Optional)

You already built RAG manually in this project. This optional section shows how **LangChain organizes the same patterns** into reusable components.

## Install

```powershell
venv\Scripts\activate
pip install -r requirements-langchain.txt
```

## Run the demo

```powershell
python langchain_demo.py              # both demos
python langchain_demo.py --mode chain # LCEL RAG chain only
python langchain_demo.py --mode react # ReAct agent only
```

This script shows two LangChain patterns side by side — without changing `app.py` or `rag_pipeline.py`:

1. **LCEL chain** — always retrieve, then generate (`retriever | prompt | llm`)
2. **ReAct agent** — the model decides when to call a retriever tool in a Thought → Action → Observation loop

## Concept mapping

| Your manual code | LangChain equivalent |
|------------------|----------------------|
| `data_loader.py` | `Document` objects |
| `embeddings.py` | `HuggingFaceEmbeddings` |
| `vector_store.py` | `Chroma` + `.as_retriever()` |
| Prompt string in `generate_answer()` | `ChatPromptTemplate` |
| `_client.models.generate_content()` | `ChatGoogleGenerativeAI` |
| `run_rag()` orchestration | LCEL chain (`retriever \| prompt \| llm \| parser`) |
| Agent-style retrieval | ReAct agent via `langgraph.prebuilt.create_react_agent` + `@tool` |
| `conversation.py` | `MessagesPlaceholder` / memory classes (not shown in demo) |
| `security.py` | Custom runnable or middleware (you still own this) |
| `monitoring.py` | Callbacks / LangSmith tracing (optional in production) |
| `filters.py` | Retriever config or post-retrieval filtering |
| `workflow.py` | Query transformation runnables before retriever |

## ReAct agent support

The ReAct demo uses:

- **`langgraph.prebuilt.create_react_agent`** — builds the agent graph (LLM ↔ tools loop)
- **`@tool` from `langchain_core.tools`** — wraps the Chroma retriever as `search_knowledge_base`
- **`ChatGoogleGenerativeAI`** — same Gemini model as the main app

Flow:

```text
User question
  → LLM decides: answer directly or call search_knowledge_base
  → Tool runs retriever.invoke(query)
  → LLM reads tool result and continues until done
```

LangChain 1.x also offers `langchain.agents.create_agent` as the newer agent factory. This project uses `create_react_agent` because it maps directly to the classic **ReAct** (Reason + Act) pattern from the course.

## What LangChain helps with

- **Structure** — chains compose steps clearly
- **Reusability** — swap models, retrievers, or prompts with less boilerplate
- **Team scale** — common patterns are easier to share

## What LangChain does NOT do

- It does **not** add new AI capabilities
- It does **not** remove the need for security, filtering, or evaluation
- It can **hide details** — debugging may be harder until you understand the abstractions

## Questions to consider (from assignment)

1. Does LangChain simplify your code or obscure it?
2. Where do you lose control compared to `rag_pipeline.py`?
3. Is debugging easier or harder?
4. When would this help on a team project?

## Files

| File | Purpose |
|------|---------|
| `langchain_demo.py` | Standalone LangChain RAG demo |
| `requirements-langchain.txt` | Optional LangChain dependencies |
| `chroma_db_langchain/` | Separate vector store for the demo (auto-created) |
