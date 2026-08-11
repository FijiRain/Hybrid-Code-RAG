import json
from typing import List

from .data_models import MinimalSource


def iou(a_start: int, a_end: int,
        b_start: int, b_end: int) -> float:
    """Compute the overlap ratio between two source spans.

    Args:
        a_start: Start offset of the first span.
        a_end: End offset of the first span.
        b_start: Start offset of the second span.
        b_end: End offset of the second span.

    Returns:
        The intersection-over-union score between the two spans.
    """

    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    if intersection == 0:
        return 0.0

    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union


def k_len(sources: list[MinimalSource]) -> int:
    """Return the number of retrieved sources in a result list.

    Args:
        sources: Source entries used to compute the list length.

    Returns:
        The number of sources contained in the list.
    """
    return len(sources)


def calculate_iou(answer_path: str,
                  dataset_path: str) -> tuple[float, int]:
    """Compute the mean recall score for retrieved sources.

    Args:
        answer_path: Path to the answer dataset.
        dataset_path: Path to the retrieval dataset.

    Returns:
        A tuple with the mean recall value and the retrieval parameter k.
    """
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
