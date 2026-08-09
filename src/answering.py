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
    prompt = (
            "Answer the question in a complete, natural sentence, using only "
            "the information in the context below.\n\n"
            "Example:\n"
            "Context: [...]\n"
            "Question: Where can I find information about X?\n"
            "ANSWER: Information about X can be found in "
            "the X documentation page.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "ANSWER:"
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
