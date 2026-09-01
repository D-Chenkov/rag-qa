# rag-qa - Project #2 (RAG document Q&A)

Ask questions about your own PDFs and get answers grounded in them, with source citations. A retrieval-augmented generation (RAG) system running **fully local** (Ollama + FAISS, no API keys). Portfolio Project #2 of my QA-Automation to ML-Engineer transition.

STATUS: working - local RAG + conversation memory + RAGAS evaluation. Retrieval tuned via controlled experiments (see Evaluation).

## Pipeline (the six steps)

```
INGEST (once):   load PDF -> split into chunks -> embed -> store in FAISS
QUERY (per ask):                         retrieve top-k -> generate grounded answer
```

- **load** (`PyPDFLoader`) -> **split** (`RecursiveCharacterTextSplitter`, overlapping chunks) -> **embed** (`OllamaEmbeddings`, nomic-embed-text) -> **store** (`FAISS`, saved to disk). See `src/ingest.py`.
- **retrieve** (embed the question, similarity search, top-k) -> **generate** (stuff chunks into the prompt, `ChatOllama` answers only from context). See `src/rag.py`.

## Stack
Local + free: **Ollama** (LLM `gemma4:e4b`, embeddings `nomic-embed-text`), **FAISS** vector store, **LangChain** orchestration, **FastAPI** serving, **RAGAS** for eval. No OpenAI key needed.

## Structure
```
rag-qa/
  data/               PDFs (gitignored) -> put your docs here
  faiss_index/        saved index (gitignored, built by ingest.py)
  src/
    config.py         model names, chunk size, top-k, paths
    ingest.py         steps 1-4: load -> split -> embed -> store
    rag.py            steps 5-6: retrieve -> generate (grounded answer + sources); dense or hybrid
    chat.py           conversational RAG (condense follow-up -> retrieve -> answer, with memory)
    agent.py          agentic RAG (LangGraph StateGraph; retriever as a tool + checkpointer memory)
    eval_set.py       15 fixed, committed ground-truth Q/A pairs (the eval set)
    eval_ragas.py     RAGAS eval (faithfulness, answer relevancy, context precision/recall)
  app/main.py         FastAPI: POST /ask, GET /health
  tests/test_app.py   pytest (/health)
  requirements.txt
```

## Quickstart
```bash
# prerequisites: Ollama running, models pulled
ollama pull gemma4:e4b
ollama pull nomic-embed-text

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# add PDFs to data/, then build the index
python src/ingest.py

# ask on the CLI
python src/rag.py "What does the document say about X?"

# or serve it
uvicorn app.main:app --port 8080
curl -X POST localhost:8080/ask -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

## Evaluation (RAGAS)

Retrieval and generation are scored separately with RAGAS (LLM-as-judge, using the local Ollama model). The eval set is 15 **fixed, committed** ground-truth Q/A pairs (`src/eval_set.py`), kept constant across experiments so score deltas are attributable to the *system change*, not the questions. Metrics: `context_recall` + `context_precision` (retrieval), `faithfulness` + `answer_relevancy` (generation).

**Controlled experiments** (same 15-question set, same judge, one knob changed at a time):

| config | context_recall | context_precision | faithfulness | answer_relevancy |
|--------|:--:|:--:|:--:|:--:|
| dense, k=8, chunk=1000 (baseline) | **0.533** | **0.375** | **0.500** | **0.359** |
| + hybrid (BM25 + dense, 50/50) | 0.400 | 0.286 | 0.467 | 0.308 |
| + smaller chunks (500, k=8) | 0.267 | 0.236 | 0.333 | 0.173 |

**Findings:**
- **Hybrid retrieval hurt on this corpus.** BM25 (exact-keyword, bag-of-words) adds noise on literary prose, which has few distinctive terms; its common-word matches displaced good semantic hits in the RRF fusion. BM25 pays off on keyword-heavy/technical corpora, not narrative text.
- **Smaller chunks hurt at fixed `k`.** With `TOP_K` fixed, halving chunk size halves the *total* text retrieved, so recall collapsed. Shrinking chunks needs a larger `k` to keep coverage constant.
- **Winner: the dense baseline.** Both attempted "improvements" regressed - a concrete reminder to measure, not assume.

**Caveats (honest):** the local `gemma4:e4b` judge is a rough grader and n=15 is small, so absolute values are *directional* - trust only clear, consistent moves (e.g. dense beating hybrid on all four metrics), not sub-0.05 wobble. Next levers if revisited (from Anthropic's Contextual Retrieval): contextual chunks, reranking, a stronger embedding model.

Reproduce: `python src/eval_ragas.py` (flip `config.RETRIEVER` / `CHUNK_SIZE` to rerun an experiment).

## Roadmap
- [x] Repo scaffold (ingest + rag + serving, local stack)
- [x] Build the index on a real PDF and get grounded answers
- [x] Conversation memory (condense -> retrieve -> answer, `src/chat.py`)
- [x] RAGAS evaluation on a fixed eval set; controlled retrieval experiments
- [x] Hybrid (BM25 + dense) retriever - tried, measured, dense won on this corpus
- [x] LangGraph agent variant (`src/agent.py`): retriever-as-tool, decides when to search, checkpointer memory
- [ ] Deploy on a cloud VM + pin
- [ ] Dockerfile + a small UI

## Notes
LangChain's import paths shift between versions - if an import breaks, check the current docs; the concepts (loader, splitter, embeddings, vector store, retriever, chain) are stable. Answers are only as good as retrieval: if the model says "I don't know," inspect the retrieved chunks first (that's a retrieval problem, not a generation one).
