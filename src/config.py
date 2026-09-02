"""Central config so the LLM/embeddings/retrieval knobs live in one place."""

import os

# Local stack (Ollama). Swap freely - the rest of the code is provider-agnostic.
LLM_MODEL = "gemma4:e4b"          # ollama pull gemma4:e4b
EMBED_MODEL = "nomic-embed-text"  # ollama pull nomic-embed-text

# Where Ollama is reachable. Defaults to localhost; inside a container set
# OLLAMA_HOST=http://host.docker.internal:11434 to reach Ollama on the host.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Retrieval / chunking knobs (tune these with RAGAS).
# Winning baseline from the controlled experiments (see README RAGAS table):
# dense, chunk=1000/overlap=150, k=8. Smaller chunks regressed recall at fixed k.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 8

# Retriever strategy: "dense" (FAISS only) or "hybrid" (BM25 + dense fusion).
# Flip to "hybrid" and re-run eval_ragas.py as a controlled experiment.
RETRIEVER = "dense"   # dense beat hybrid on this corpus (see README RAGAS table)

# Paths
DATA_DIR = "data"          # put your PDFs here (gitignored)
INDEX_DIR = "faiss_index"  # saved FAISS index (gitignored)
