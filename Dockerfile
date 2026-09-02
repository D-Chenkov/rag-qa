# rag-qa serving image. Ollama runs on the HOST (not in this container),
# so the LLM/embeddings weights aren't baked in and the image stays small.
FROM python:3.12-slim

WORKDIR /app

# System deps: faiss needs libgomp; build-essential covers native wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY static/ ./static/

# Reach the host's Ollama from inside the container.
# On Docker Desktop (Mac/Windows) host.docker.internal resolves automatically.
# On Linux, run with: --add-host=host.docker.internal:host-gateway
ENV OLLAMA_HOST=http://host.docker.internal:11434

EXPOSE 8080

# The FAISS index lives in a volume you mount at run time (it is gitignored and
# built by src/ingest.py), e.g.  -v $(pwd)/faiss_index:/app/faiss_index
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
