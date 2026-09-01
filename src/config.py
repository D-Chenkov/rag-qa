"""Central config so the LLM/embeddings/retrieval knobs live in one place."""

# Local stack (Ollama). Swap freely - the rest of the code is provider-agnostic.
LLM_MODEL = "gemma4:e4b"          # ollama pull gemma4:e4b
EMBED_MODEL = "nomic-embed-text"  # ollama pull nomic-embed-text

# Retrieval / chunking knobs (tune these with RAGAS)
#DEFAULTS
#CHUNK_SIZE = 1000
#CHUNK_OVERLAP = 150
#TOP_K = 4

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 8

# Retriever strategy: "dense" (FAISS only) or "hybrid" (BM25 + dense fusion).
# Flip to "hybrid" and re-run eval_ragas.py as a controlled experiment.
RETRIEVER = "dense"   # dense beat hybrid on this corpus (see README RAGAS table)

# Paths
DATA_DIR = "data"          # put your PDFs here (gitignored)
INDEX_DIR = "faiss_index"  # saved FAISS index (gitignored)
