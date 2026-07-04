import asyncio

from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

from database.connection import get_async_session
from database.vector_store import similarity_search
from tools.reranker import rerank

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embeddings: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


@tool
async def search_text_chunks(query: str, k: int = 8) -> list[dict]:
    """
    Search indexed document text. The bi-encoder fetches the candidate top-k by
    cosine from the vector store, then the cross-encoder reranks that candidate
    set for the LLM. Returns up to k chunks with content, page_number,
    similarity, rerank_score, doc_filename.

    The accept/reject relevance gate (evals/retrieval_eval.py and
    pipeline/hybrid.py's per-chunk floor) runs on rerank_score, not cosine.
    Cosine here only decides the initial top-k candidate set handed to the
    reranker. Reranking is mandatory: tools/reranker.py raises if the model
    can't be loaded or scoring fails, instead of returning cosine-only results
    — there is no fallback path that serves chunks without a rerank_score.
    """
    # embed_query and rerank are synchronous, CPU-bound model inference — run
    # them off the event loop so they don't stall other concurrent requests.
    embedding = await asyncio.to_thread(_get_embeddings().embed_query, query)
    async with get_async_session() as session:
        candidates = await similarity_search(session, embedding, k)
    return await asyncio.to_thread(rerank, query, candidates, k)
