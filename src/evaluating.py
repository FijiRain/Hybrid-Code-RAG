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
