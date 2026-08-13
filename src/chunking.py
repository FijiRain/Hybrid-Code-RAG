from tqdm import tqdm
from langchain_text_splitters import (PythonCodeTextSplitter,
                                      MarkdownTextSplitter)

from .data_models import MinimalSource


def verify_indices(files: list[tuple[str, str]],
                   sources: list[MinimalSource],
                   overlap: int) -> None:
    """Check that chunk metadata matches the original source content.

    Args:
        files: Original files as a list of (path, content) pairs.
        sources: Chunk metadata extracted from those files.
        overlap: Chunk overlap used during splitting.

    Returns:
        None. It prints the number of mismatched chunks found.
    """
    errors = 0
    for source in sources:
        # Find the matching original file
        content = next(c for path, c in files if path == source.file_path)
        extracted = content[source.first_character_index:
                            source.last_character_index]
        if extracted != source.content:
            errors += 1
            print(
                f"chunk_index={source.chunk_index}, "
                f"len(chunk)={len(source.content)}, "
                f"OVERLAP={overlap}"
            )
            print(f"[MISMATCH] {source.file_path} chunk {source.chunk_index}")
            print(f"  Expected (extracted): {extracted[:80]!r}...")
            print(f"  Got (source.content): {source.content[:80]!r}...")
    print(f"{errors} / {len(sources)} chunks mismatched")


def extract_chunk(file_path: str, content: str, chunks: list[str],
                  overlap: int) -> list[MinimalSource]:
    """Convert chunks into source metadata with offsets.

    Args:
        file_path: Path of the source file.
        content: Full original file content.
        chunks: Chunks produced by the text splitter.
        overlap: Overlap size used between adjacent chunks.

    Returns:
        A list of chunk metadata objects with file path, offsets, and content.
    """
    sources = []
    cursor = 0
    for i, chunk in enumerate(chunks):
        start_index = content.find(chunk, cursor)
        end_index = start_index + len(chunk)
        # bc if end_index < OVERLAP, cursor < 0 so next iter will fail...
        cursor = max(start_index, end_index - overlap)
        entity = MinimalSource(
                                file_path=file_path,
                                first_character_index=start_index,
                                last_character_index=end_index,
                                chunk_index=i,
                                content=chunk,
                                )
        sources.append(entity)
    return sources


def py_chunker(files: list[tuple[str, str]],
               max_chunk_size: int, overlap: int) -> list[MinimalSource]:
    """Split Python source files into overlapping chunks.

    Args:
        files: List of (file path, file content) pairs for Python files.
        max_chunk_size: Maximum size of each chunk.
        overlap: Number of overlapping characters between neighboring chunks.

    Returns:
        A list of chunk metadata objects for the Python files.
    """
    sources = []
    # Note: LangChain guaranties chunk_overlap < chunk_size
    chunker = PythonCodeTextSplitter(
                                chunk_size=max_chunk_size,
                                chunk_overlap=overlap
                                )
    for elem in tqdm(files, desc="Chunking Python files...",
                     colour="#05701C"):
        chunks = chunker.split_text(elem[1])
        sources.extend(extract_chunk(elem[0], elem[1], chunks, overlap))
    return sources


def md_chunker(files: list[tuple[str, str]],
               max_chunk_size: int, overlap: int) -> list[MinimalSource]:
    """Split Markdown files into overlapping chunks.

    Args:
        files: List of (file path, file content) pairs for Markdown files.
        max_chunk_size: Maximum size of each chunk.
        overlap: Number of overlapping characters between neighboring chunks.

    Returns:
        A list of chunk metadata objects for the Markdown files.
    """
    sources = []
    chunker = MarkdownTextSplitter(
                            chunk_size=max_chunk_size,
                            chunk_overlap=overlap
                            )
    for elem in tqdm(files, desc="Chunking Markdown files...",
                     colour="#05701C"):
        chunks = chunker.split_text(elem[1])
        sources.extend(extract_chunk(elem[0], elem[1], chunks, overlap))
    return sources
