from .data_models import (StudentSearchResults, MinimalSearchResults)

'''
class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int

class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]
'''

def jsp() -> None:
    ...