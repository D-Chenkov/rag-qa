"""Conversational RAG: multi-turn Q&A with memory.

Adds chat history ON TOP of the base pipeline without touching rag.answer()
(which the eval uses). Standard conversational-RAG flow:
  1. CONDENSE  - rewrite a follow-up into a standalone question using history
                 (so "and his sister?" becomes "Who is Gregor's sister?").
  2. RETRIEVE  - similarity search on that standalone question.
  3. ANSWER    - generate from the retrieved context + the history.

Uses only stable LCEL primitives (prompts | llm | parser) to survive LangChain
version drift. Run:  python src/chat.py
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

import config
from rag import load_retriever, _format_docs

_llm = ChatOllama(model=config.LLM_MODEL, temperature=0, base_url=config.OLLAMA_BASE_URL)

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the conversation so far and a follow-up question, rewrite the follow-up "
     "as a STANDALONE question that makes sense on its own. Only output the rewritten "
     "question - do NOT answer it."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Answer using ONLY the context below. If the answer isn't in the context, say you "
     "don't know - don't make anything up. Be concise.\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])


def chat_once(question, chat_history, retriever=None):
    """One turn. chat_history is a list of HumanMessage/AIMessage.
    Returns (answer, docs, new_history)."""
    retriever = retriever or load_retriever()

    # 1. CONDENSE - only when there's history (first turn needs no rewrite)
    if chat_history:
        standalone = (CONDENSE_PROMPT | _llm | StrOutputParser()).invoke(
            {"chat_history": chat_history, "question": question}
        )
    else:
        standalone = question

    # 2. RETRIEVE on the standalone question
    docs = retriever.invoke(standalone)

    # 3. ANSWER with context + history
    answer = (ANSWER_PROMPT | _llm | StrOutputParser()).invoke(
        {"context": _format_docs(docs), "chat_history": chat_history, "question": question}
    )

    new_history = chat_history + [HumanMessage(content=question), AIMessage(content=answer)]
    return answer, docs, new_history


def main():
    retriever = load_retriever()
    history = []
    print("Conversational RAG - ask about the indexed document (Ctrl-C to quit).")
    while True:
        try:
            q = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        answer, docs, history = chat_once(q, history, retriever)
        print(f"\nbot > {answer}")
        pages = sorted({d.metadata.get("page") for d in docs})
        print(f"      (sources: pages {pages})")


if __name__ == "__main__":
    main()
