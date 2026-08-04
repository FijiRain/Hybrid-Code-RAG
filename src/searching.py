import bm25s

from .constants import DATA_PROCESSED
from .data_models import MinimalSource
from .utils import reload_chunks


def search_sources(query: str, k: int) -> list[MinimalSource]:
    query_tokens = bm25s.tokenize(query)
    all_chunks = reload_chunks()
    retriever = bm25s.BM25.load(DATA_PROCESSED)
    results = retriever.retrieve(query_tokens, all_chunks, k)

    return [
        MinimalSource.model_validate(item)
        for item in results[0][0]
    ]