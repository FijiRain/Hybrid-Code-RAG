import bm25s
from rich import print
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np

from .constants import (DEF_MAX_CHUNK_SIZE, DATA_INDEX,
                        SAVE_SEARCH, DATA_DOCS, ANSWER_DOCS,
                        SEARCH_DOCS, SEARCH_AND_ANSWER, SEM_MODEL,
                        SEM_CHUNKS, DATA_CODE, SEARCH_CODE, ANSWER_CODE)
from .data_models import (RagDataset, MinimalSearchResults,
                          StudentSearchResults, StudentSearchResultsAndAnswer,
                          MinimalAnswer, SemanticSearchResult,
                          StudentSemanticSearchResults)
from .indexing import build_index, build_embeddings_index
from .searching import search_sources, semantic_search, reciprocal_rank_fusion
from .answering import generate_simple_answer, load_model
from .evaluating import calculate_iou
from .utils import reload_chunks, save_searchresults, thinking_length


class Cli():
    """CLI entry point for indexing, searching, and evaluating RAG data."""

    def index(self, max_chunk_size: int = DEF_MAX_CHUNK_SIZE) -> None:
        """Build the retrieval index from raw source files.

        Args:
            max_chunk_size: Maximum length of each chunk used for indexing.

        Returns:
            None.
        """
        build_index(max_chunk_size)

    def search(self, query: str, k: int = 5, r_type: str = "lex") -> None:
        """Search the indexed corpus and print the top matching sources.

        Args:
            query: User query to search for.
            k: Number of top results to return.

        Returns:
            None.
        """
        if not query:
            raise ValueError("Query cannnot be empty.")
        if k <= 0:
            raise ValueError("k must be > 0.")

        all_chunks = reload_chunks()
        if r_type == "lex":
            retriever = bm25s.BM25.load(DATA_INDEX)
            results = search_sources(query, k, all_chunks, retriever)
            for source in results:
                print(
                    f"{source.file_path} "
                    f"[{source.first_character_index}:"
                    f"{source.last_character_index}]"
                )

        # --- Bonus part ---
        elif r_type == "sem":
            model = SentenceTransformer(SEM_MODEL)
            chunks_content = [c["content"] for c in all_chunks]
            emb_chunks = model.encode(chunks_content)
            np.save(SEM_CHUNKS, emb_chunks)
            s_results = semantic_search(model, query, k,
                                        all_chunks, emb_chunks)
            for src in s_results:
                print(f"{src.file_path} [Score: {src.score}]")
        # ------------------

    def search_dataset(self, dataset_path: str = DATA_DOCS,
                       save_directory: str = SAVE_SEARCH,
                       k: int = 5,
                       r_type: str = "lex") -> None:
        """Run retrieval over a dataset and save the search results.

        Args:
            dataset_path: Path to the dataset containing unanswered questions.
            save_directory: Directory where results should be written.
            k: Number of top sources to keep per question.

        Returns:
            None.
        """
        try:
            with open(dataset_path, "r") as f:
                dataset = RagDataset.model_validate_json(f.read())
        except FileNotFoundError:
            raise FileNotFoundError("Incorrect dataset_path value.")
        if k <= 0:
            raise ValueError("k must be > 0.")

        all_chunks = reload_chunks()
        if r_type == "lex":
            retriever = bm25s.BM25.load(DATA_INDEX)
            minsearch_list = []
            for uq in tqdm(dataset.rag_questions, desc="Retrieving...",
                           colour="green"):
                top_sources = search_sources(uq.question, k,
                                             all_chunks, retriever)
                ms_result = MinimalSearchResults(question_id=uq.question_id,
                                                 question=uq.question,
                                                 retrieved_sources=top_sources)
                minsearch_list.append(ms_result)

            result = StudentSearchResults(search_results=minsearch_list, k=k)
            filename = Path(dataset_path).name
            print("[bold green]Dataset lexical search done![/bold green]")
            save_searchresults(result, save_directory, filename)
            print(f"[green]Saved student_search_results "
                  f"to {save_directory}{filename}[/green]")

        # --- Bonus Part ---
        elif r_type == "sem":
            search_list = []
            model = SentenceTransformer(SEM_MODEL)
            emb_chunks = build_embeddings_index(model, all_chunks)
            for uq in tqdm(dataset.rag_questions, desc="Score generation",
                           colour="green"):
                s_sources = semantic_search(model, uq.question,
                                            k, all_chunks, emb_chunks)
                sm_result = SemanticSearchResult(question_id=uq.question_id,
                                                 question=uq.question,
                                                 retrieved_sources=s_sources)
                search_list.append(sm_result)
            s_result = StudentSemanticSearchResults(search_results=search_list,
                                                    k=k)
            filename = "sem_" + Path(dataset_path).name
            print("[bold green]Dataset semantic search done![/bold green]")
            save_searchresults(s_result, save_directory, filename)
            print(f"[green]Saved student_search_results "
                  f"to {save_directory}{filename}[/green]")
        # -------------------

    def answer(self, query: str, k: int = 5) -> None:
        """Answer a single question using the retrieval pipeline.

        Args:
            query: User question to answer.
            k: Number of source chunks to retrieve before answering.

        Returns:
            None. The answer is printed to stdout.
        """
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
        """Generate answers for every question in a search-results dataset.

        Args:
            student_search_results_path: Path to the saved search results file.
            save_directory: Directory where the answer file will be written.

        Returns:
            None.
        """

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
        """Evaluate retrieval recall against a reference dataset.

        Args:
            student_search_results_path: Path to the retrieved results.
            dataset_path: Reference dataset used to compute recall.

        Returns:
            None. The score is printed to stdout.
        """
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
        """Measure generation time on a small diagnostic set.

        Returns:
            None.
        """
        thinking_length()

    def retrieval_pipeline(self, dataset_path1: str = DATA_DOCS,
                           dataset_path2: str = DATA_CODE,
                           save_directory: str = SAVE_SEARCH,
                           student_search_results_path1: str = SEARCH_DOCS,
                           student_search_results_path2: str = SEARCH_CODE,
                           answer_path1: str = ANSWER_DOCS,
                           answer_path2: str = ANSWER_CODE,
                           k: int = 5
                           ) -> None:
        """Run the full retrieval pipeline.

        Args:
            dataset_path: Dataset containing questions to retrieve.
            save_directory: Directory where search results are written.
            student_search_results_path: Path to saved retrieval results.
            answer_path: Reference answer dataset for evaluation.
            k: Number of top sources to keep per question.

        Returns:
            None.
        """
        self.index()
        print()
        self.search_dataset(dataset_path1, save_directory, k)
        print()
        self.search_dataset(dataset_path2, save_directory, k)
        print()
        self.evaluate(student_search_results_path1, answer_path1)
        print()
        self.evaluate(student_search_results_path2, answer_path2)

    # --- Bonus part ---
    def hybrid_retrieval(self, lex_dataset: str,
                         sem_dataset: str) -> None:
        """Fuse lexical and semantic search results using RRF.

        Args:
            lex_dataset: Path to the lexical search results dataset.
            sem_dataset: Path to the semantic search results dataset.

        Raises:
            FileNotFoundError: If either dataset cannot be found.
            ValueError: If datasets contain different numbers of questions or
                mismatched question IDs.
        """
        try:
            with open(lex_dataset, "r") as f:
                lex = StudentSearchResults.model_validate_json(f.read())
        except FileNotFoundError:
            raise FileNotFoundError("Incorrect dataset_path value.")
        try:
            with open(sem_dataset, "r") as f:
                sem = (StudentSemanticSearchResults.
                       model_validate_json(f.read()))
        except FileNotFoundError:
            raise FileNotFoundError("Incorrect dataset_path value.")

        if len(lex.search_results) != len(sem.search_results):
            raise ValueError("Lexical and semantic datasets "
                             "must have the same number of questions.")

        target_k = min(lex.k, sem.k)

        hybrid_results = []
        for lex_res, sem_res in zip(lex.search_results, sem.search_results):
            if lex_res.question_id != sem_res.question_id:
                raise ValueError("Question IDs do not match "
                                 "between lexical and semantic datasets.")

            fused_sources = reciprocal_rank_fusion(
                lex_res.retrieved_sources,
                sem_res.retrieved_sources
            )[:target_k]
            hybrid_results.append(MinimalSearchResults(
                question_id=lex_res.question_id,
                question=lex_res.question,
                retrieved_sources=fused_sources
            ))

        result = StudentSearchResults(search_results=hybrid_results,
                                      k=target_k)
        filename = "hybrid_" + Path(lex_dataset).name
        save_searchresults(result, SAVE_SEARCH, filename)
        print(f"[green]Saved hybrid_search_results to "
              f"{SAVE_SEARCH}{filename}[/green]")
    # ------------------
