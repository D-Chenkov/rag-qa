"""FastAPI serving for the RAG Q&A system.

POST /ask {"question": "..."} -> {"answer": "...", "sources": [...]}
Needs a built FAISS index (run src/ingest.py) and Ollama running.
"""

import os
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from rag import answer, load_retriever

app = FastAPI(title="rag-qa")
_retriever = None


class Query(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(q: Query):
    global _retriever
    try:
        if _retriever is None:
            _retriever = load_retriever()      # lazy: build once on first question
        text, docs = answer(q.question, _retriever)
    except Exception as e:                      # e.g. missing index / Ollama down
        raise HTTPException(status_code=503, detail=f"RAG not ready: {e}")
    sources = [{"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in docs]
    return {"answer": text, "sources": sources}
