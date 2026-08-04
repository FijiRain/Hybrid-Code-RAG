from rich import print
import bm25s

from .constants import DEF_MAX_CHUNK_SIZE, DATA_PROCESSED
from .utils import reload_chunks
from .data_models import RagDataset, MinimalSearchResults, StudentSearchResults
from .indexing import build_index


class Cli():
    def index(self, max_chunk_size: int = DEF_MAX_CHUNK_SIZE) -> None:
        build_index(max_chunk_size)

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
                  f"{i["last_character_index"]}]") # retravailler si refactor, on retournera une liste de MinimalSource a la place

    def search_dataset(self, dataset_path: str, k: int,
                       save_directory: str) -> None:
        try:
            with open(dataset_path, "r") as f:
                dataset = RagDataset.model_validate_json(f.read())
        except FileNotFoundError:
            raise FileNotFoundError("Incorrect dataset_path value.")
        if k < 0:
            raise ValueError("k must be > 0.")

        results_list = []
        for uq in dataset.rag_questions:
            top_sources = []  # remplacer par une fonction : func(uq.question, k) et implementer comme le search en haut, refactor possible
            question_tokens = bm25s.tokenize(uq.question)  #  Est-ce que ca va pas poser probleme parce que le tokenizer n'est pas au meme endroit que le tokenizer du corpus?
            # retrieval ici
            result = MinimalSearchResults(question_id=uq.question_id,
                                          question=uq.question,
                                          retrieved_sources=top_sources)
            results_list.append(result)

        result = StudentSearchResults(search_results=results_list, k=k)
        try:
            with open(save_directory, "w") as f:
                ...
        except FileNotFoundError:
            raise FileNotFoundError("Incorrect save_directory value.")

    def answer(self, query: str, k: int) -> None:
        ...

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        ...

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        ...
