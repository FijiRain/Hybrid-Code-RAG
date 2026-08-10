import json
from typing import List

from .data_models import MinimalSource


def iou(a_start: int, a_end: int,
        b_start: int, b_end: int) -> float:

    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    if intersection == 0:
        return 0.0

    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union


def k_len(sources: list[MinimalSource]) -> int:
    return len(sources)


def calculate_iou(answer_path: str,
                  dataset_path: str) -> tuple[float, int]:
    with open(answer_path, "r") as f:
        answers = json.load(f)
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    recall_lst: List[float] = []
    k = 0
    for a_question in answers["rag_questions"]:
        for d_question in dataset["search_results"]:
            if a_question["question_id"] == d_question["question_id"]:
                nb_source_find = 0
                for a_source in a_question["sources"]:
                    found = False
                    k = k_len(d_question["retrieved_sources"])
                    for b_source in d_question["retrieved_sources"][:k]:
                        if (a_source["file_path"] == b_source["file_path"]
                            and
                            (iou(a_source["first_character_index"],
                                 a_source["last_character_index"],
                                 b_source["first_character_index"],
                                 b_source["last_character_index"],
                                 )) >= 0.05):
                            # Bc the source overlap is sufficent
                            found = True
                            break
                    if found:
                        nb_source_find += 1
                recall_lst.append(nb_source_find / len(a_question["sources"]))
    return (sum(recall_lst) / len(recall_lst), k)

# """Answer generation module for RAG system.

# This module provides functions for generating answers using a language model
# based on retrieved chunks for questions.
# """

# from .Index import Index
# from .config import (
#     GPU_UTILISATION,
#     K,
#     MAX_MODEL_LEN,
#     MAX_NUM_SEQ,
#     MAX_TOKEN,
#     TEMPERATURE,
#     TOP_P,
#     VLLM_FILE,
#     MODEL_NAME,
# )
# from .pydantic_validator import (
#     MinimalAnswer,
#     MinimalSearchResults,
#     MinimalSource,
#     StudentSearchResults,
#     StudentSearchResultsAndAnswer,
# )
# from .querry import get_question, search_best_chunks

# import json
# import os
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple
# from tqdm import tqdm # type: ignore[import-untyped]
# from transformers import AutoTokenizer, PreTrainedTokenizerBase

# os.environ["VLLM_LOGGING_LEVEL"] = "ERROR" # noqa
# from vllm import LLM, SamplingParams# noqa


# def init_model() -> Tuple[LLM, SamplingParams]:
#     """Initialize language model and sampling parameters.

#     Returns:
#         Tuple of (LLM model, SamplingParams).
#     """
#     model = LLM(
#         model=MODEL_NAME,
#         max_model_len=MAX_MODEL_LEN,
#         gpu_memory_utilization=GPU_UTILISATION,
#         max_num_seqs=MAX_NUM_SEQ,
#         disable_log_stats=True,
#     )

#     params = SamplingParams(
#         temperature=TEMPERATURE,
#         top_p=TOP_P,
#         max_tokens=MAX_TOKEN,
#         stop=["<|im_end|>", "<|endoftext|>"],
#     )
#     return (model, params)


# file_cache: Dict[str, str] = {}


# def get_chunks_str(retrieved_sources: List[Dict[str, Any]]) -> List[str]:
#     """Extract chunk content from files.

#     Args:
#         retrieved_sources: List of source information dictionaries.

#     Returns:
#         List of chunk content strings.
#     """
#     chunks_str = []
#     for chunk in retrieved_sources:
#         file = chunk["file_path"]

#         if file not in file_cache:
#             with open(file, "r") as fd:
#                 file_cache[file] = fd.read()
#         file_data = file_cache[file]
#         start = chunk["first_character_index"]
#         end = chunk["last_character_index"]
#         chunks_str.append(file_data[start:end])
#     return chunks_str


# def verif_tokens_len(tokenizer: PreTrainedTokenizerBase, context: str) -> bool:
#     """Verify context doesn't exceed token limit.

#     Args:
#         tokenizer: Tokenizer instance.
#         context: Context string to check.

#     Returns:
#         True if context is within token limit, False otherwise.
#     """
#     prompt_tokens = len(tokenizer.encode(context))
#     if prompt_tokens > 7500:
#         print(f"Prompt too long: {prompt_tokens} tokens")
#         return False
#     return True


# def gen_dataset_result(
#     path_question: str, path_answer: str, k: int = K
# ) -> Dict[str, Any]:
#     """Generate search results for all questions in a dataset.

#     Args:
#         path_question: Path to question file.
#         path_answer: Path to save results.
#         k: Number of top chunks to retrieve. Defaults to K.

#     Returns:
#         Dictionary containing search results.
#     """
#     from .cli import loading # noqa: F401

#     dataset = get_question(path_question)
#     search_lst: List[MinimalSearchResults] = []
#     results = []

#     index = Index()
#     index.load(VLLM_FILE)
#     for i, question in enumerate(
#         tqdm(
#             dataset.rag_questions,
#             desc="Search dataset",
#             bar_format="{desc} {bar} {percentage:.0f}%",
#         )
#     ):
#         search_lst.append(search_best_chunks(index, question, k))

#         sources = []
#         for source in search_lst[i].retrieved_sources:
#             sources.append(
#                 {
#                     "file_path": source.file_path,
#                     "first_character_index": source.first_character_index,
#                     "last_character_index": source.last_character_index,
#                 }
#             )

#         result_dict = {
#             "question_id": question.question_id,
#             "question_str": question.question,
#             "retrieved_sources": sources,
#         }
#         results.append(result_dict)

#     output = {"search_results": results, "k": k}

#     with open(path_answer, "w") as f:
#         json.dump(output, f, indent=4)

#     return output


# def get_result_students_answer(
#     path_answer: str, path_save: str, k: int = K
# ) -> Tuple[StudentSearchResults, StudentSearchResultsAndAnswer]:
#     """Generate answers for all questions in search results.

#     Args:
#         path_answer: Path to search results file.
#         path_save: Path to save answers.
#         k: Number of top chunks. Defaults to K.

#     Returns:
#         Tuple of (StudentSearchResults, StudentSearchResultsAndAnswer).

#     Raises:
#         ValueError: If dataset file not found.
#     """
#     from .cli import suppress_output # noqa: F401

#     with suppress_output():
#         model, params_model = init_model()
#     tokenizer: PreTrainedTokenizerBase = (
#         AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
#     )

#     if not Path(path_answer).exists():
#         raise ValueError("No dataset found")
#     with open(path_answer, "r") as f:
#         data = json.load(f)

#     search_lst: List[MinimalSearchResults] = []
#     anwser_lst: List[MinimalAnswer] = []
#     for question in tqdm(
#         data["search_results"],
#         desc="Gen answer",
#         bar_format="{desc} {bar} {percentage:.0f}%",
#     ):
#         chunks_str = get_chunks_str(question["retrieved_sources"])

#         context = (
#             "<|im_start|>system\n"
#             "You are a helpful assistant. "
#             "Read the following code chunks carefully and answer the "
#             "question directly. "
#             "Do not repeat the instructions. Do not say 'the answer is in "
#             "the chunks'. "
#             "Just give the answer.\n"
#             f"Chunks:\n{chr(10).join(f'--- Chunk {i+1} ---{chr(10)}{c}'
#                                      for i, c in enumerate(chunks_str))}\n"
#             "<|im_end|>\n"
#             "<|im_start|>user\n"
#             f"{question['question_str']}\n"
#             "<|im_end|>\n"
#             "<|im_start|>assistant\n"
#             "<think>\n\n</think>\n"
#         )

#         if not verif_tokens_len(tokenizer, context):
#             continue

#         search_lst.append(
#             MinimalSearchResults(
#                 question_id=question["question_id"],
#                 question=question["question_str"],
#                 retrieved_sources=[
#                     MinimalSource(**source)
#                     for source in question["retrieved_sources"]
#                 ],
#             )
#         )

#         with suppress_output():
#             outputs = model.generate(context, params_model)

#         question.update({"answer": outputs[0].outputs[0].text})

#         sources: List[MinimalSource] = []
#         for source in question["retrieved_sources"]:
#             sources.append(
#                 MinimalSource(
#                     file_path=source["file_path"],
#                     first_character_index=source["first_character_index"],
#                     last_character_index=source["last_character_index"],
#                 )
#             )
#         anwser_lst.append(
#             MinimalAnswer(
#                 question_id=question["question_id"],
#                 question=question["question_str"],
#                 retrieved_sources=sources,
#                 answer=outputs[0].outputs[0].text,
#             )
#         )

#     with open(path_save, "w") as f:
#         json.dump(data, f, indent=4)

#     student_search: StudentSearchResults = StudentSearchResults(
#         search_results=search_lst, k=k
#     )
#     student_search_answer: StudentSearchResultsAndAnswer = (
#         StudentSearchResultsAndAnswer(search_results=anwser_lst, k=k)
#     )

#     return (student_search, student_search_answer)


# def gen_single_question_result(
#     question_str: str, question_id: str, k: int
# ) -> Dict[str, Any]:
#     """Generate search results for a single question.

#     Args:
#         question_str: Question string.
#         question_id: Question ID.
#         k: Number of top chunks to retrieve.

#     Returns:
#         Dictionary containing search results.
#     """
#     from .cli import loading # noqa: F401

#     @dataclass
#     class SimpleQuestion:
#         """Simple question class for searching."""

#         question: str
#         question_id: str

#     search_lst: List[MinimalSearchResults] = []
#     results = []

#     question = SimpleQuestion(question=question_str, question_id=question_id)

#     index = Index()
#     index.load(VLLM_FILE)
#     search_lst.append(search_best_chunks(index, question, k)) # type: ignore

#     sources = []
#     for source in search_lst[0].retrieved_sources:
#         sources.append(
#             {
#                 "file_path": source.file_path,
#                 "first_character_index": source.first_character_index,
#                 "last_character_index": source.last_character_index,
#             }
#         )

#     result_dict = {
#         "question_id": question_id,
#         "question_str": question_str,
#         "retrieved_sources": sources,
#     }
#     results.append(result_dict)

#     output = {"search_results": results, "k": k}

#     return output


# def get_result_student_answer_single(
#     question_str: str, question_id: str, k: int = K
# ) -> Tuple[Optional[MinimalSearchResults], Optional[MinimalAnswer]]:
#     """Generate answer for a single question.

#     Args:
#         question_str: Question string.
#         question_id: Question ID.
#         k: Number of top chunks. Defaults to K.

#     Returns:
#         Tuple of (MinimalSearchResults, MinimalAnswer) or (None, None) if
#         prompt too long.
#     """
#     from .cli import loading, suppress_output # noqa: F401

#     with suppress_output():
#         model, params_model = init_model()
#     tokenizer: PreTrainedTokenizerBase = (
#         AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
#     )

#     search_result = loading(
#         gen_single_question_result,
#         "Search on dataset",
#         question_str,
#         question_id,
#         k,
#     )

#     search_lst: List[MinimalSearchResults] = []
#     anwser_lst: List[MinimalAnswer] = []

#     for question in tqdm(
#         search_result["search_results"],
#         desc="Gen single answer",
#         bar_format="{desc} {bar} {percentage:.0f}%",
#     ):
#         chunks_str = get_chunks_str(question["retrieved_sources"])

#         context = (
#             "<|im_start|>system\n"
#             "You are a helpful assistant. "
#             "Read the following code chunks carefully and answer the "
#             "question directly. "
#             "Do not repeat the instructions. Do not say 'the answer is in "
#             "the chunks'. "
#             "Just give the answer.\n"
#             f"Chunks:\n{chr(10).join(f'--- Chunk {i+1} ---{chr(10)}{c}'
#                                      for i, c in enumerate(chunks_str))}\n"
#             "<|im_end|>\n"
#             "<|im_start|>user\n"
#             f"{question['question_str']}\n"
#             "<|im_end|>\n"
#             "<|im_start|>assistant\n"
#             "<think>\n\n</think>\n"
#         )

#         if not verif_tokens_len(tokenizer, context):
#             return None, None

#         search_lst.append(
#             MinimalSearchResults(
#                 question_id=question["question_id"],
#                 question=question["question_str"],
#                 retrieved_sources=[
#                     MinimalSource(**source)
#                     for source in question["retrieved_sources"]
#                 ],
#             )
#         )

#         with suppress_output():
#             outputs = model.generate(context, params_model)

#         sources: List[MinimalSource] = []
#         for source in question["retrieved_sources"]:
#             sources.append(
#                 MinimalSource(
#                     file_path=source["file_path"],
#                     first_character_index=source["first_character_index"],
#                     last_character_index=source["last_character_index"],
#                 )
#             )
#         anwser_lst.append(
#             MinimalAnswer(
#                 question_id=question["question_id"],
#                 question=question["question_str"],
#                 retrieved_sources=sources,
#                 answer=outputs[0].outputs[0].text,
#             )
#         )

#     return search_lst[0], anwser_lst[0]