# LangChain CLI — uses the same pipeline as streamlit run app.py
#
# Usage:
#   python langchain_demo.py "What is Python?"
#   python langchain_demo.py   # interactive prompts

import sys

from rag_pipeline import initialize_vector_store, run_rag


def main():
    print("Initializing LangChain vector store...")
    doc_count = initialize_vector_store()
    print(f"Knowledge base ready ({doc_count} documents).\n")

    questions = sys.argv[1:] or [
        "What is Python?",
        "What else can it do in the real world?",
    ]

    for question in questions:
        print(f"Q: {question}")
        result = run_rag(question)
        if result["error"]:
            print(f"A: [error] {result['answer']}\n")
            continue
        print(f"A: {result['answer']}")
        if result["sources"]:
            print(f"   ({len(result['sources'])} sources, confidence {result['confidence']:.0%})")
        print()


if __name__ == "__main__":
    main()
