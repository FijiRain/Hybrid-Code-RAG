from rich import print

from .constants import (DEF_MAX_CHUNK_SIZE, DATA_PROCESSED)
from .chunking import py_chunker, md_chunker
from .utils import load_files


class Cli():
    def index(self, max_chunk_size: int = DEF_MAX_CHUNK_SIZE) -> None:
        if max_chunk_size < 0 or max_chunk_size > 2000:
            raise ValueError("--max_chunk_size range is 0 to 2000 max.")

        files = load_files("data/raw/vllm-0.10.1")
        py_files = [f for f in files if f[0].endswith(".py")]
        md_files = [f for f in files if f[0].endswith(".md")]

        py_chunks = py_chunker(py_files, max_chunk_size)
        md_chunks = md_chunker(md_files, max_chunk_size)


        # for s in py_chunks:
        #     print(s)
        # mettre le BM25 ici qu'on soumet a TQDM
        # print((
        #     f"[bold green]Ingestion complete! Indexed "
        #     f"{len(py_chunks + md_chunks)} chunks from "
        #     f"{len(files)} files in {DATA_PROCESSED}."
        #     "[/bold green]"
        #     ))
