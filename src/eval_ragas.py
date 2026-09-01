"""RAGAS evaluation for the RAG pipeline (RAGAS 0.4.x API).

Flow (crawl before you walk):
  1. A small hand-written eval set: question + reference (ground-truth) answer.
  2. Run OUR rag.answer() on each question -> response + retrieved_contexts.
  3. Assemble a RAGAS EvaluationDataset from those rows.
  4. evaluate() with a LOCAL Ollama judge (LLM + embeddings).

Prerequisites: Ollama running, models pulled, and the FAISS index built
(python src/ingest.py). Run from the repo root:  python src/eval_ragas.py

Metric class names occasionally shift between RAGAS versions. If an import
below fails on your 0.4.3 install, list what's available with:
    python -c "import ragas.metrics as m; print([n for n in dir(m) if n[0].isupper()])"
"""

# --- Compatibility shim (RAGAS 0.4.3 vs LangChain 1.x) ---
# ragas/llms/base.py eagerly imports ChatVertexAI from the OLD path
# `langchain_community.chat_models.vertexai`, which was removed in
# langchain-community 1.x. We don't use Vertex, so stub it so RAGAS imports.
import sys as _sys
import types as _types
try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    import langchain_community.chat_models as _cm
    _stub = _types.ModuleType("langchain_community.chat_models.vertexai")
    _stub.ChatVertexAI = object  # placeholder, never used (Ollama is the judge)
    _sys.modules["langchain_community.chat_models.vertexai"] = _stub
    setattr(_cm, "vertexai", _stub)
# --- end shim ---

from langchain_ollama import ChatOllama, OllamaEmbeddings

from ragas import EvaluationDataset, evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,                     # answer grounded in retrieved context?
    ResponseRelevancy,                # answer addresses the question? (needs embeddings)
    LLMContextRecall,                 # did retrieval fetch what's needed? (needs reference)
    LLMContextPrecisionWithReference, # are retrieved chunks relevant? (needs reference)
)

import config
from rag import answer          # our pipeline: answer(q) -> (text, docs)
from eval_set import EVAL_SET    # fixed, committed ground-truth Q/A


def build_dataset():
    """Step 2 + 3: run our RAG on each question, collect rows RAGAS expects."""
    rows = []
    for item in EVAL_SET:
        q = item["question"]
        response, docs = answer(q)                     # our pipeline
        rows.append({
            "user_input": q,
            "retrieved_contexts": [d.page_content for d in docs],
            "response": response,
            "reference": item["reference"],
        })
    return EvaluationDataset.from_list(rows)


def main():
    dataset = build_dataset()

    # local judge (temp 0 for a stable grader)
    judge_llm = LangchainLLMWrapper(ChatOllama(model=config.LLM_MODEL, temperature=0))
    judge_emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=config.EMBED_MODEL))

    # A single local Ollama serves calls ~serially, so don't fire 16 at once:
    # low max_workers avoids contention (each call runs faster), high timeout
    # lets the slow LLM-heavy metrics (faithfulness, context precision) finish.
    run_config = RunConfig(timeout=600, max_workers=2)

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextRecall(),
            LLMContextPrecisionWithReference(),
        ],
        llm=judge_llm,
        embeddings=judge_emb,
        run_config=run_config,
    )
    print(result)
    #Faithfulness, answer_relevancy, context_recall, llm_context_precision_with_reference
    return result


if __name__ == "__main__":
    main()

# --- Next (flywheel step #2): synthetic test data ---
# Instead of hand-writing EVAL_SET, generate it from the corpus with RAGAS's
# TestsetGenerator (docs: getstarted/rag_testset_generation). Get this eval
# version green first, THEN swap in generation.
