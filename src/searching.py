import bm25s
import math
import numpy as np
import torch
from typing import Any, Sequence
from sentence_transformers import SentenceTransformer

from .data_models import MinimalSource, SemanticSource


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
        similarity.append((c, float(cos_sim)))
    res = sorted(similarity, key=lambda x: x[1], reverse=True)
    return res[:k]


def semantic_search(model: SentenceTransformer,
                    query: str,
                    k: int,
                    all_chunks: list[dict[str, Any]],
                    emb_chunks: torch.Tensor) -> list[SemanticSource]:
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
    sims = find_similarities(emb_query, emb_chunks, all_chunks, k)
    return [SemanticSource.model_validate({**chunk, "score": score})
            for chunk, score in sims]


def reciprocal_rank_fusion(
    lex_dataset: Sequence[MinimalSource | dict[str, Any]],
    sem_dataset: Sequence[SemanticSource |
                          tuple[dict[str, Any], float] | dict[str, Any]]
                        ) -> list[MinimalSource]:
    scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict[str, Any]] = {}
    k = 60

    for rank, chunk in enumerate(lex_dataset):
        if isinstance(chunk, MinimalSource):
            chunk_dict = chunk.model_dump()
        else:
            chunk_dict = chunk

        key = f"{chunk_dict['file_path']}:{chunk_dict['chunk_index']}"
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        chunk_lookup[key] = chunk_dict

    for rank, sem_item in enumerate(sem_dataset):
        # Accept chunk or (chunk, score)
        chunk = sem_item[0] if isinstance(sem_item, tuple) else sem_item
        if isinstance(chunk, SemanticSource):
            chunk_dict = chunk.model_dump()
        else:
            chunk_dict = chunk

        key = f"{chunk_dict['file_path']}:{chunk_dict['chunk_index']}"
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        chunk_lookup[key] = chunk_dict

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [MinimalSource.model_validate(chunk_lookup[key])
            for key, _ in ranked]
# ------------------
