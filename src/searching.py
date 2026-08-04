import bm25s
from typing import Any

from .data_models import MinimalSource


def search_sources(query: str, k: int,
                   all_chunks: list[dict[str, Any]],
                   retriever: bm25s.BM25) -> list[MinimalSource]:
    query_tokens = bm25s.tokenize(query)
    results = retriever.retrieve(query_tokens, all_chunks, k)

    return [MinimalSource.model_validate(item) for item in results[0][0]]
