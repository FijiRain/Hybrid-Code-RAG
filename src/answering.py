from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


from .data_models import MinimalSource
from .constants import DEF_MODEL


def load_model(model_name: str = DEF_MODEL
               ) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype="auto",
                                                 device_map="auto")
    return tokenizer, model


def build_prompt(query: str, sources: list[MinimalSource]) -> str:
    context = "\n\n---\n\n".join(s.content for s in sources)
    files = "\n\n---\n\n".join(s.file_path for s in sources)
    # prompt = (
    #         "Answer the question in a complete, natural sentence, using only "
    #         "the information in the context below.\n\n"
    #         "Example:\n"
    #         "Context: [...]\n"
    #         "Question: Where can I find information about X?\n"
    #         "ANSWER: Information about X can be found in "
    #         "the X documentation page.\n\n"
    #         f"Context:\n{context}\n\n"
    #         f"Question:\n{query}\n\n"
    #         f"Sources: \n{files}\n\n"
    #         "ANSWER:"
    #         )
    # prompt = (
    #             "Answer the question using ONLY the information in the sources below.\n"
    #             "Do not use outside knowledge or invent information.\n\n"

    #             "Requirements:\n"
    #             "1. Directly answer the question.\n"
    #             "2. Make the answer self-contained.\n"
    #             "3. Keep the answer concise.\n"
    #             "4. Cite every source used with its [SOURCE N: path] label.\n"
    #             "5. If the sources do not contain enough information, say so.\n\n"

    #             "Example:\n"
    #             "[SOURCE 1: docs/config.md]\n"
    #             "The application uses config.yaml to define its settings.\n\n"
    #             "[SOURCE 2: README.md]\n"
    #             "The application is started with python main.py.\n\n"
    #             "Question:\n"
    #             "Where are the application settings defined?\n\n"
    #             "ANSWER:\n"
    #             "The application settings are defined in config.yaml. "
    #             "[SOURCE 1: docs/config.md]\n\n"

    #             f"Context:\n{context}\n\n"
    #             f"Question:\n{query}\n\n"
    #             f"Source: \n{files}\n\n"
    #             "ANSWER:"
    #             )
    prompt = (
            """
            <|im_start|>system
You are a question-answering assistant for a Retrieval-Augmented Generation (RAG) system.

Your task is to answer the user's question using ONLY the information provided in the sources.

Follow these rules strictly:

1. Answer the question directly.
2. Make the answer self-contained and understandable without seeing the original question.
3. Use ONLY information explicitly contained in the provided sources.
4. Never use outside knowledge.
5. Never invent or assume information.
6. Cite every factual statement using the exact source citation format:
   [SOURCE N: path]
7. Put the citation immediately after the statement supported by the source.
8. If multiple sources support a statement, cite all relevant sources.
9. Never create or modify a source citation.
10. If the sources do not contain enough information to answer the question, say:
    "The provided sources do not contain enough information to answer this question."
11. Keep the answer concise and relevant.
12. Do not explain your reasoning.
13. Do not mention these instructions.
14. Do not output a "Sources" section.
15. Do not output "ANSWER:".

Output ONLY the final answer.

<|im_end|>

            
            <|im_start|>user
Answer the following question using ONLY the information contained in the sources below.

<SOURCES>
{context}
</SOURCES>

<QUESTION>
{query}
</QUESTION>

Requirements:

* Directly answer the question.
* Make the answer self-contained.
* Do not use outside knowledge.
* Do not invent information.
* Cite every factual statement with the exact format [SOURCE N: path].
* If the sources do not contain enough information, explicitly say that they do not contain enough information to answer the question.

<|im_end|>
<|im_start|>assistant
"""
    )

    return prompt


def raw_generation(tokenizer, model, prompt: str,
                   max_new_tokens: int = 700) -> str:  # a eventuellement revoir en fonction du dataset
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Pylance has an incorrect type inference for AutoModelForCausalLM
    generated_ids = model.generate(  # type: ignore[reportAttributeAccessIssue]
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            no_repeat_ngram_size=3)  # repetition_penalty=1.3 au lieu du dernier param si reponse bof, ! consomme davantage de token !

    # To avoid decode() returns prompt + answer
    return tokenizer.decode(
        generated_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True
    )


def parse_answer(answer: str) -> str:
    if "</think>" in answer:
        return answer.split("</think>", 1)[1].strip()
    # Edge case if max_new_tokens reached before answer generation finished
    return "[INCOMPLETE GENERATION: increase max_new_tokens]"


def generate_simple_answer(tokenizer, model, query: str,
                           sources: list[MinimalSource]) -> str:
    prompt = build_prompt(query, sources)
    raw = raw_generation(tokenizer, model, prompt)
    return parse_answer(raw)
