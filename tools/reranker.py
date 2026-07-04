"""
Cross-encoder reranking — a precision stage on top of vector retrieval.

The bi-encoder (all-MiniLM) retrieves candidates fast but with coarse scores:
genuinely relevant and clearly-absent chunks sit only ~0.06 apart in cosine
space, which makes a threshold fragile. A cross-encoder jointly encodes the
(query, chunk) pair and produces a much sharper relevance score, so both the
ordering of context and the accept/reject gate become more reliable.

The accept/reject gates (evals/retrieval_eval.py, pipeline/hybrid.py) are
calibrated on rerank_score alone — there is no cosine-based fallback. If the
reranker can't be loaded or fails, that's a hard failure: it propagates to the
caller rather than silently serving less-precise, uncalibrated results.
"""
import logging
import math
import os

from sentence_transformers import CrossEncoder

# Swap in any sentence-transformers cross-encoder via env (e.g. a larger/BGE
# reranker). Note: Qwen3-Reranker is NOT a CrossEncoder — it's a CausalLM scored
# on the 'yes' token and needs a different code path + a GPU (see notes below).
_MODEL_NAME = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
_model: CrossEncoder | None = None
log = logging.getLogger("pipeline.reranker")


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """
    Re-score (query, chunk) pairs with the cross-encoder and return the top_k by
    relevance. Attaches `rerank_score` (0–1, sigmoid of the model logit) to each
    chunk. No fallback: if the model can't be loaded or scoring fails, the
    exception propagates — callers must not serve results gated on a threshold
    that was never calibrated for cosine similarity.
    """
    if not chunks:
        return chunks
    model = _get_model()
    scores = model.predict([(query, c["content"]) for c in chunks])

    for chunk, score in zip(chunks, scores, strict=True):
        chunk["rerank_score"] = _sigmoid(float(score))
    chunks.sort(key=lambda c: c["rerank_score"], reverse=True)
    return chunks[:top_k]
