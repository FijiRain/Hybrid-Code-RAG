from rich import print
import bm25s
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from .chunking import md_chunker, py_chunker, verify_indices
from .constants import (DATA_CHUNKS, DATA_INDEX, DATA_RAW,
                        DEF_MAX_CHUNK_SIZE, SEM_CHUNKS)
from .utils import dump_chunks, load_files


def build_index(max_chunk_size: int = DEF_MAX_CHUNK_SIZE) -> None:
    """Build and persist the BM25 index from the raw corpus.

    Args:
        max_chunk_size: Maximum number of characters allowed per chunk.

    Returns:
        None.
    """
    if max_chunk_size < 0 or max_chunk_size > DEF_MAX_CHUNK_SIZE:
        raise ValueError("--max_chunk_size range is 0 to 2000 max.")

    # Empirically based on 20% of chunk size for recall@k score
    overlap = (max_chunk_size * 20) // 100

    files = load_files(DATA_RAW)
    py_files = [f for f in files if f[0].endswith(".py")]
    md_files = [f for f in files if f[0].endswith(".md")]

    # py file, md files, max_chunk, overlap
    py_chunks = py_chunker(py_files, max_chunk_size, overlap)
    md_chunks = md_chunker(md_files, max_chunk_size, overlap)
    py_data_dict = [ms.model_dump(mode="json") for ms in py_chunks]
    md_data_dict = [ms.model_dump(mode="json") for ms in md_chunks]
    dump_chunks(py_data_dict, md_data_dict)

    # Combine all chunks in ONE corpus
    # Otherwise we'll have to calculate separate indexes py/md and mix them
    # Bc IDF will not be calculated on the same basis
    all_chunks = py_chunks + md_chunks
    print()

    print("[yellow]Indexes' reliability verification:[/yellow]")
    verify_indices(files, all_chunks, overlap)
    print()

    all_content = [c.content for c in all_chunks]
    # "content" used to build the index
    corpus_tokens = bm25s.tokenize(all_content)

    # All dict stored in the retriever
    all_chunks_dict = py_data_dict + md_data_dict
    retriever = bm25s.BM25(corpus=all_chunks_dict)
    retriever.index(corpus_tokens)
    retriever.save(DATA_INDEX)

    print((
        f"[bold green]Ingestion complete! Indexed "
        f"{len(all_chunks)} chunks from "
        f"{len(files)} files in {DATA_CHUNKS}."
        "[/bold green]"))


def build_embeddings_index(model: SentenceTransformer,
                           all_chunks: list[dict]) -> torch.Tensor:
    chunks_content = [c["content"] for c in all_chunks]
    embeddings: torch.Tensor = model.encode(chunks_content)
    np.save(SEM_CHUNKS, embeddings)
    return embeddings
