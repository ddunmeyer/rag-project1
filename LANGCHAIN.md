# Week 15.5 — LangChain Introduction (Optional)

You already built RAG manually in this project. This optional section shows how **LangChain organizes the same patterns** into reusable components.

## Install

```powershell
venv\Scripts\activate
pip install -r requirements-langchain.txt
```

## Run the demo

```powershell
python langchain_demo.py
```

This script rebuilds one simple flow — **user question → retrieve docs → generate answer** — without changing `app.py` or `rag_pipeline.py`.

## Concept mapping

| Your manual code | LangChain equivalent |
|------------------|----------------------|
| `data_loader.py` | `Document` objects |
| `embeddings.py` | `HuggingFaceEmbeddings` |
| `vector_store.py` | `Chroma` + `.as_retriever()` |
| Prompt string in `generate_answer()` | `ChatPromptTemplate` |
| `_client.models.generate_content()` | `ChatGoogleGenerativeAI` |
| `run_rag()` orchestration | LCEL chain (`retriever \| prompt \| llm \| parser`) |
| `conversation.py` | `MessagesPlaceholder` / memory classes (not shown in demo) |
| `security.py` | Custom runnable or middleware (you still own this) |
| `monitoring.py` | Callbacks / LangSmith tracing (optional in production) |
| `filters.py` | Retriever config or post-retrieval filtering |
| `workflow.py` | Query transformation runnables before retriever |

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
