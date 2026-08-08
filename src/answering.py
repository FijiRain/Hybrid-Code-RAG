from .data_models import MinimalSource

def generate_simple_answer(query: str, sources: list[MinimalSource], k: int) -> None:
    context = "\n\n---\n\n".join(s.content for s in sources)
    prompt = (
            "You are a question-answering assistant. "
            "Answer the question based exclusively on the provided context. "
            "Do not use outside knowledge. Do not invent or assume "
            "information that is not present in the context. If the context "
            "does not contain "
            "enough information to answer the question, reply exactly: "
            "'I don't have enough information to answer this question.'\n\n"
            "Context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{query}\n\n"
            "Use exactly this final output format:\n"
            "ANSWER: <your answer>\n"
            )