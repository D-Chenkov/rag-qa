"""RAG steps 1-4: LOAD -> SPLIT -> EMBED -> STORE.

Builds a FAISS index from the PDFs in data/. Run once (re-run when docs change):
    python src/ingest.py

Note: LangChain's import paths drift between versions - if an import fails,
check the current docs. Concepts (loader, splitter, embeddings, vector store)
are stable; the package names move.
"""

import glob
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

import config


def load_chunks(data_dir=config.DATA_DIR):
    """Steps 1-2: LOAD every PDF -> SPLIT into overlapping chunks.
    Reusable: FAISS uses these to embed; BM25 (hybrid retrieval) uses them raw.
    Fast (no embeddings), so it's cheap to call at retriever-build time too."""
    docs = []
    for pdf in glob.glob(os.path.join(data_dir, "*.pdf")):
        docs += PyPDFLoader(pdf).load()
    if not docs:
        raise SystemExit(f"No PDFs found in {data_dir}/ - add some and re-run.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


def build_index(data_dir=config.DATA_DIR, index_dir=config.INDEX_DIR):
    chunks = load_chunks(data_dir)                        # steps 1-2
    # 3. EMBED + 4. STORE - vectorize each chunk and index it in FAISS
    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_dir)
    print(f"indexed {len(chunks)} chunks -> {index_dir}/")


if __name__ == "__main__":
    build_index()
