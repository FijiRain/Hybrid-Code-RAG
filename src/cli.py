from rich import print
import bm25s

from .constants import (DEF_MAX_CHUNK_SIZE, DATA_PROCESSED,
                        DATA_RAW, DATA_CHUNKS)
from .chunking import py_chunker, md_chunker
from .utils import load_files, dump_chunks, reload_chunks
from .chunking import verify_indices


class Cli():
    def index(self, max_chunk_size: int = DEF_MAX_CHUNK_SIZE) -> None:
        if max_chunk_size < 0 or max_chunk_size > DEF_MAX_CHUNK_SIZE:
            raise ValueError("--max_chunk_size range is 0 to 2000 max.")

        # Empirically based on 10% of chunk size for recall@k score
        overlap = max_chunk_size // 10

        files = load_files(DATA_RAW)
        py_files = [f for f in files if f[0].endswith(".py")]
        md_files = [f for f in files if f[0].endswith(".md")]

        py_chunks = py_chunker(py_files, max_chunk_size, overlap)
        md_chunks = md_chunker(md_files, max_chunk_size, overlap)
        py_data_dict = [ms.model_dump(mode="json") for ms in py_chunks]
        md_data_dict = [ms.model_dump(mode="json") for ms in md_chunks]
        dump_chunks(py_data_dict, md_data_dict)
        # Combine all chunks in ONE corpus
        # Otherwise we'll have to calculate separate index (py/md) and mix them
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
        retriever.save(DATA_PROCESSED)

        print((
            f"[bold green]Ingestion complete! Indexed "
            f"{len(all_chunks)} chunks from "
            f"{len(files)} files in {DATA_CHUNKS}."
            "[/bold green]"))

    def search(self, query: str, k: int) -> None:
        if not query:
            raise ValueError("Query cannnot be empty.")
        if k < 0:
            raise ValueError("k must be > 0.")

        query_tokens = bm25s.tokenize(query)
        all_chunks = reload_chunks()
        retriever = bm25s.BM25.load(DATA_PROCESSED)
        results = retriever.retrieve(query_tokens, all_chunks, k)
        for i in results[0][0]:
            print(f"{i["file_path"]} "
                  f"[{i["first_character_index"]}:"
                  f"{i["last_character_index"]}]")

    def search_dataset(self, dataset_path: str, k: int,
                       save_directory: str) -> None:
        ...

    def answer(self, query: str, k: int) -> None:
        ...

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        ...

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        ...
