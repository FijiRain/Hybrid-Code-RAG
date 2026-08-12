import bm25s
import math
import numpy as np
import torch
from typing import Any
from sentence_transformers import SentenceTransformer

from .data_models import MinimalSource


def search_sources(query: str, k: int,
                   all_chunks: list[dict[str, Any]],
                   retriever: bm25s.BM25) -> list[MinimalSource]:
    """Retrieve the top matching source chunks for a query with BM25.

    Args:
        query: Search query submitted by the user.
        k: Number of top results to retrieve.
        all_chunks: Chunk payloads indexed in the corpus.
        retriever: BM25 retriever containing the indexed corpus.

    Returns:
        The list of source metadata matching the query, ordered by relevance.
    """
    query_tokens = bm25s.tokenize(query)
    results = retriever.retrieve(query_tokens, all_chunks, k)

    return [MinimalSource.model_validate(item) for item in results[0][0]]


# --- Bonus part ---
def find_similarities(query_embedding: torch.Tensor,
                      chunks_embedding: torch.Tensor,
                      all_chunks: list[dict[str, Any]],
                      k: int) -> list[tuple[dict, float]]:
    """Return the top k chunks by cosine similarity.

    Args:
        query_embedding: Embedding of the query vector.
        chunks_embedding: Embeddings of the candidate chunks.
        all_chunks: Chunk payloads matched to the embeddings.
        k: Number of highest-similarity chunks to keep.

    Returns:
        The k best chunk-score pairs ordered by descending similarity.
    """

    similarity = []
    norm_query = math.sqrt(sum(v ** 2 for v in query_embedding))
    for c, e in zip(all_chunks, chunks_embedding):
        norm_c = math.sqrt(sum(v ** 2 for v in e))
        dot = np.dot(query_embedding, e)
        cos_sim = (dot)/(norm_query * norm_c)
        similarity.append((c, cos_sim))
    res = sorted(similarity, key=lambda x: x[1], reverse=True)
    return res[:k]


def semantic_search(model: SentenceTransformer,
                    query: str,
                    k: int,
                    all_chunks: list[dict[str, Any]],
                    emb_chunks:
                    torch.Tensor) -> list[tuple[dict[str, Any], float]]:
    """Search semantically similar chunks for a query.

    Args:
        model: SentenceTransformer used to encode text.
        query: User query to compare against chunks.
        k: Number of closest chunks to return.
        all_chunks: Chunk payloads to compare.
        emb_chunks: Unused embedding parameter kept for compatibility.

    Returns:
        The top k chunk-score pairs ranked by cosine similarity.
    """
    emb_query = model.encode(query)
    chunks_content = [c["content"] for c in all_chunks]
    emb_chunks = model.encode(chunks_content)

    sims = find_similarities(emb_query, emb_chunks, all_chunks, k)
    return sims
# ------------------
