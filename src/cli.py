import bm25s
from rich import print
from pathlib import Path

from .constants import (DEF_MAX_CHUNK_SIZE, DATA_INDEX,
                        SAVE_SEARCH, DATA_DOCS, ANSWER_DOCS, SEARCH_DOCS)
from .data_models import RagDataset, MinimalSearchResults, StudentSearchResults
from .indexing import build_index
from .searching import search_sources
from .answering import generate_simple_answer, load_model
from .evaluating import calculate_iou
from .utils import reload_chunks, save_searchresults, thinking_length

'''
copypasta

index           : uv run python -m src index
search          : uv run python -m src search "How to configure the OpenAI server?" --k 5
search dataset  : uv run python -m src search_dataset --dataset_path datasets_public/public/UnansweredQuestions/dataset_code_public.json
moulinette      : ./moulinette evaluate_student_search_results output/search_results/UnansweredQuestions/dataset_docs_public.json datasets_public/public/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000

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

    def search(self, query: str, k: int = 5) -> None:
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

    def search_dataset(self, dataset_path: str = DATA_DOCS,
                       save_directory: str = SAVE_SEARCH,
                       k: int = 5) -> None:
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

    def answer(self, query: str, k: int = 5) -> None:
        if not query:
            raise ValueError("Query cannnot be empty.")
        if k <= 0:
            raise ValueError("k must be > 0.")

        all_chunks = reload_chunks()
        retriever = bm25s.BM25.load(DATA_INDEX)
        search_res = search_sources(query, k, all_chunks, retriever)
        tokenizer, model = load_model()
        answer_text = generate_simple_answer(tokenizer, model,
                                             query, search_res)
        print(answer_text)

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        # TODO
        ...

    def evaluate(self, student_search_results_path: str = SEARCH_DOCS,
                 dataset_path: str = ANSWER_DOCS) -> None:
        res, k = calculate_iou(dataset_path, student_search_results_path)
        if res >= 0.8 and "docs" in student_search_results_path:
            print(f"[green]Your Recall@{k} score for docs is: "
                  f"{res*100:.1f}%![/green]")
        elif res >= 0.5 and "code" in student_search_results_path:
            print(f"[green]Your Recall@{k} score for code is: "
                  f"{res*100:.1f}%![/green]")
        else:
            print(f"[red]Insufficent Recall@{k}, score : {res*100:.1f}%[/red]")

    # Extra CLI helper commands
    def diagnose_thinking_length(self) -> None:
        thinking_length()

    def all_pipeline(self, dataset_path: str = DATA_DOCS,
                     save_directory: str = SAVE_SEARCH,
                     student_search_results_path: str = SEARCH_DOCS,
                     answer_path: str = ANSWER_DOCS
                     ) -> None:
        self.index()
        self.search_dataset(dataset_path, save_directory)
        self.evaluate(student_search_results_path, answer_path)
