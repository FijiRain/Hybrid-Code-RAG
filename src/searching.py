import bm25s
from typing import Any

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
