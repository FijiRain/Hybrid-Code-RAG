import bm25s
from rich import print
from pathlib import Path

from .constants import DEF_MAX_CHUNK_SIZE, DATA_INDEX
from .data_models import RagDataset, MinimalSearchResults, StudentSearchResults
from .indexing import build_index
from .searching import search_sources
from .answering import generate_simple_answer
from .utils import reload_chunks, save_searchresults

'''
copypasta

index           : uv run python -m src index
search          : uv run python -m src search
                  "How to configure the OpenAI server?" --k 5
search dataset  : uv run python -m src search_dataset
                  --dataset_path
                  datasets_public/public/UnansweredQuestions/
                  dataset_docs_public.json --k 10
                  --save_directory
                  data/output/search_results/UnansweredQuestions
moulinette      : ./moulinette evaluate_student_search_results
                  output/search_results/UnansweredQuestions/
                  UnansweredQuestions.json
                  datasets_public/public/AnsweredQuestions/
                  dataset_code_public.json --k 10
                  --max_context_length 2000

answer_dataset  : uv run python -m src answer_dataset
                  --student_search_results_path
                  data/output/search_results/UnansweredQuestions/
                  dataset_docs_public.json
                  --save_directory
                  data/output/search_results_and_answer/
                  UnansweredQuestions
'''


class Cli():
    def index(self, max_chunk_size: int = DEF_MAX_CHUNK_SIZE) -> None:
        build_index(max_chunk_size)

    def search(self, query: str, k: int) -> None:
        if not query:
            raise ValueError("Query cannnot be empty.")
        if k <= 0:
            raise ValueError("k must be > 0.")

        all_chunks = reload_chunks()
        retriever = bm25s.BM25.load(DATA_INDEX)
        results = search_sources(query, k, all_chunks, retriever)
        for source in results:
            print(
                f"{source.file_path} "
                f"[{source.first_character_index}:"
                f"{source.last_character_index}]"
            )

    def search_dataset(self, dataset_path: str, k: int,
                       save_directory: str) -> None:
        try:
            with open(dataset_path, "r") as f:
                dataset = RagDataset.model_validate_json(f.read())
        except FileNotFoundError:
            raise FileNotFoundError("Incorrect dataset_path value.")
        if k <= 0:
            raise ValueError("k must be > 0.")

        all_chunks = reload_chunks()
        retriever = bm25s.BM25.load(DATA_INDEX)
        minsearch_list = []
        for uq in dataset.rag_questions:
            top_sources = search_sources(uq.question, k, all_chunks, retriever)
            ms_result = MinimalSearchResults(question_id=uq.question_id,
                                             question=uq.question,
                                             retrieved_sources=top_sources)
            minsearch_list.append(ms_result)

        result = StudentSearchResults(search_results=minsearch_list, k=k)
        filename = Path(dataset_path).name
        save_searchresults(result, save_directory, filename)
        print(f"[bold green]Saved student_search_results "
              f"to {save_directory}/{filename}[/bold green]")

    def answer(self, query: str, k: int) -> None:
        if not query:
            raise ValueError("Query cannnot be empty.")
        if k <= 0:
            raise ValueError("k must be > 0.")

        all_chunks = reload_chunks()
        retriever = bm25s.BM25.load(DATA_INDEX)
        search_res = search_sources(query, k, all_chunks, retriever)
        answer = generate_simple_answer(query, search_res, k)

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        ...

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        ...
