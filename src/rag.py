"""RAG steps 5-6: RETRIEVE relevant chunks -> GENERATE a grounded answer.

Loads the FAISS index built by ingest.py, retrieves the top-k chunks for a
question, stuffs them into the prompt, and asks the local LLM to answer only
from that context.

    python src/rag.py "What does the document say about X?"
"""

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config

PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer the question using ONLY the context below. "
    "If the answer is not in the context, say you don't know - do not make anything up.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)


def load_retriever(index_dir=config.INDEX_DIR, k=config.TOP_K):
    """Dense (FAISS) by default. Set config.RETRIEVER='hybrid' for BM25+dense
    fusion (EnsembleRetriever) - a controlled experiment: flip one flag, re-eval."""
    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)
    # allow_dangerous_deserialization: FAISS index is our own local file
    vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    dense = vectorstore.as_retriever(search_kwargs={"k": k})

    if getattr(config, "RETRIEVER", "dense") != "hybrid":
        return dense

    # HYBRID = dense (semantic) + BM25 (keyword), fused by Reciprocal Rank Fusion.
    # BM25 is in-memory, so we rebuild it from the same chunks (splitting is cheap).
    # Needs: pip install rank_bm25
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    from ingest import load_chunks

    bm25 = BM25Retriever.from_documents(load_chunks())
    bm25.k = k
    return EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])


def _format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def answer(question, retriever=None):
    retriever = retriever or load_retriever()
    llm = ChatOllama(model=config.LLM_MODEL, temperature=0)   # low temp -> faithful, grounded
    docs = retriever.invoke(question)                          # step 5: RETRIEVE
    chain = PROMPT | llm | StrOutputParser()                   # step 6: GENERATE
    text = chain.invoke({"context": _format_docs(docs), "question": question})
    return text, docs
    # TODO (memory): swap to a history-aware chain for multi-turn chat


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is this document about?"
    text, docs = answer(q)
    print(text)
    print("\n--- sources ---")
    for d in docs:
        print(f"- {d.metadata.get('source')} (page {d.metadata.get('page')})")
