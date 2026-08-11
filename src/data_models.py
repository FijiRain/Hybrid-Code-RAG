from pydantic import BaseModel, Field
from typing import List
import uuid


class MinimalSource(BaseModel):
    """Metadata describing one chunk extracted from a source file.

    Attributes:
        file_path: Path to the original source file.
        first_character_index: Start offset of the chunk in the file.
        last_character_index: End offset of the chunk in the file.
        content: Content of the chunk.
        chunk_index: Index of the chunk within the file.
    """

    file_path: str
    first_character_index: int
    last_character_index: int
    content: str
    chunk_index: int


class UnansweredQuestion(BaseModel):
    """Question without a known answer or source references.

    Attributes:
        question_id: Unique identifier for the question.
        question: Question text.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Question paired with its answer and supporting source chunks.

    Attributes:
        sources: Source chunks used to answer the question.
        answer: Answer produced for the question.
    """

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Dataset grouping questions for retrieval and answer generation.

    Attributes:
        rag_questions: List of answered or unanswered questions.
    """

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Search result metadata for one question.

    Attributes:
        question_id: Identifier of the question.
        question: Question text.
        retrieved_sources: Top source chunks returned by the retriever.
    """

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Retrieval result enriched with the generated answer text.

    Attributes:
        answer: Final textual answer produced for the question.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Collection of search results for a batch of questions.

    Attributes:
        search_results: Search results associated with each question.
        k: Number of retrieved sources kept per question.
    """

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Collection of answer-enriched search results for a batch of questions.

    Attributes:
        search_results: Answered entries for the retrieved questions.
        k: Number of retrieved sources kept per question.
    """

    search_results: List[MinimalAnswer]
    k: int
