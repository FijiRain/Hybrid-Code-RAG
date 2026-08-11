import bm25s
from rich import print
from pathlib import Path
from tqdm import tqdm

from .constants import (DEF_MAX_CHUNK_SIZE, DATA_INDEX,
                        SAVE_SEARCH, DATA_DOCS, ANSWER_DOCS,
                        SEARCH_DOCS, SEARCH_AND_ANSWER)
from .data_models import (RagDataset, MinimalSearchResults,
                          StudentSearchResults, StudentSearchResultsAndAnswer,
                          MinimalAnswer)
from .indexing import build_index
from .searching import search_sources
from .answering import generate_simple_answer, load_model
from .evaluating import calculate_iou
from .utils import reload_chunks, save_searchresults, thinking_length


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
        print("[bold green]Dataset search done![/bold green]")
        save_searchresults(result, save_directory, filename)
        print(f"[green]Saved student_search_results "
              f"to {save_directory}/{filename}[/green]")

    def answer(self, query: str, k: int = 5) -> None:
        if not query:
            raise ValueError("Query cannnot be empty.")
        if k <= 0:
            raise ValueError("k must be > 0.")

        all_chunks = reload_chunks()
        retriever = bm25s.BM25.load(DATA_INDEX)
        tokenizer, model = load_model()
        search_res = search_sources(query, k, all_chunks, retriever)
        answer_text = generate_simple_answer(tokenizer, model,
                                             query, search_res)
        print(answer_text)

    def answer_dataset(self, student_search_results_path: str = SEARCH_DOCS,
                       save_directory: str = SEARCH_AND_ANSWER) -> None:

        student_results = StudentSearchResults.model_validate_json(
            Path(student_search_results_path).read_text())
        tokenizer, model = load_model()

        answers = []
        for res in tqdm(student_results.search_results,
                        desc="Generating answers...", colour="green"):
            answer_text = generate_simple_answer(
                tokenizer, model, res.question, res.retrieved_sources)
            answers.append(MinimalAnswer(
                question_id=res.question_id,
                question=res.question,
                retrieved_sources=res.retrieved_sources,
                answer=answer_text))

        output = StudentSearchResultsAndAnswer(search_results=answers,
                                               k=student_results.k)

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        output_file = save_path / f"{Path(student_search_results_path)
                                     .stem}_answers.json"
        output_file.write_text(output.model_dump_json(indent=2))

        print(f"[bold green]Generated {len(answers)} answers, "
              f"saved to {output_file}[/bold green]")

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

    def retrieval_pipeline(self, dataset_path: str = DATA_DOCS,
                           save_directory: str = SAVE_SEARCH,
                           student_search_results_path: str = SEARCH_DOCS,
                           answer_path: str = ANSWER_DOCS
                           ) -> None:
        self.index()
        print()
        self.search_dataset(dataset_path, save_directory)
        print()
        self.evaluate(student_search_results_path, answer_path)
