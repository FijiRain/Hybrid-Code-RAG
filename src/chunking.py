from langchain_text_splitters import PythonCodeTextSplitter
from tqdm import tqdm

from .data_models import MinimalSource
from .constants import OVERLAP

# def verify_indices(files: list[tuple[str, str]], sources: list[MinimalSource]) -> None:
#     errors = 0
#     for source in sources:
#         # Retrouve le fichier original correspondant
#         content = next(c for path, c in files if path == source.file_path)
#         extracted = content[source.first_character_index:source.last_character_index]
#         if extracted != source.content:
#             errors += 1
#             print(f"chunk_index={source.chunk_index}, len(chunk)={len(source.content)}, OVERLAP={OVERLAP}")
#             print(f"[MISMATCH] {source.file_path} chunk {source.chunk_index}")
#             print(f"  Expected (extracted): {extracted[:80]!r}...")
#             print(f"  Got (source.content): {source.content[:80]!r}...")
#     print(f"{errors} / {len(sources)} chunks mismatched")


def py_chunker(files: list[tuple[str, str]],
               max_chunk_size: int) -> list[MinimalSource]:
    sources = []
    # Note: LangChain guaranties chunk_overlap < chunk_size
    chunker = PythonCodeTextSplitter(
                                chunk_size=max_chunk_size,
                                chunk_overlap=OVERLAP
                                )
    for elem in tqdm(files, desc="Chunking Python files...", colour="green"):
        chunks = chunker.split_text(elem[1])
        cursor = 0
        for i, chunk in enumerate(chunks):
            start_index = elem[1].find(chunk, cursor)
            end_index = start_index + len(chunk)
            # bc if end_index < OVERLAP, cursor < 0 so next iter will fail...
            cursor = max(start_index, end_index - OVERLAP)
            entity = MinimalSource(file_path=elem[0],
                                   first_character_index=start_index,
                                   last_character_index=end_index,
                                   chunk_index=i,
                                   content=chunk)
            sources.append(entity)
    return sources

# ABCDEFGHIJKLMNOPQRSTUVWXYZ 5 et 2
# S  C E
# 0  3 5
#    S  C E
#    3  6 8
# ABCDE DEFGH GHIJK JKLMN MNOPQ PQRST STUVW VWXYZ

def md_chunker(files: list[tuple[str, str]],
               max_chunk_size: int) -> list[MinimalSource]:
    # print(files[0])
    ...
