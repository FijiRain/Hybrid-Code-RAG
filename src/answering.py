from pathlib import Path
from typing import Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding
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
    # For Qwen to be more precise for sources, let's give him just the filename
    context = "\n\n".join(
        f"[SOURCE: {Path(source.file_path).name}]\n{source.content}"
        for source in sources
    )
    prompt = (
        "Answer the question in a complete, self-contained sentence, "
        "using only the information in the context below."
        # "Your answer must explicitly name the source file it draws from "
        # "so the last element of [SOURCE] label in 'Context'.\n\n"
        "Example 1 (documentation source):\n"
        "Context: [SOURCE: generative_models.md]\n...\n"
        "Question: Where can I find information about X?\n"
        "ANSWER: Information about X can be found in generative_models.md, "
        "which documents how to use generative models.\n\n"
        "Example 2 (code source):\n"
        "Context: [SOURCE: sampler.py]\n...\n"
        "Question: What method does class Y provide for Z?\n"
        "ANSWER: According to sampler.py, "
        "class Y provides the z() method for Z.\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        "ANSWER:"
        )
    return prompt


def raw_generation(tokenizer: PreTrainedTokenizerBase,
                   model: PreTrainedModel,
                   prompt: str,
                   max_new_tokens: int = 700) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = cast(str, tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    ))
    encoded_inputs = cast(BatchEncoding,
                          tokenizer([text], return_tensors="pt"))
    inputs = cast(BatchEncoding, encoded_inputs.to(model.device))

    generated_ids = cast(
        torch.Tensor,
        cast(Any, model).generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.3,
        ),
    )
    # replace repetition_penalty=1.3 w/ no_repeat_ngram_size=3 if regression

    # To avoid decode() returns prompt + answer
    res = cast(
        str,
        tokenizer.decode(
            generated_ids[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True,
        ),
    )
    return res


def parse_answer(answer: str) -> str:
    if "</think>" in answer:
        return answer.split("</think>", 1)[1].strip()
    # Edge case if max_new_tokens reached before answer generation finished
    return "[INCOMPLETE GENERATION: increase max_new_tokens]"


def generate_simple_answer(tokenizer: PreTrainedTokenizerBase,
                           model: PreTrainedModel,
                           query: str,
                           sources: list[MinimalSource]) -> str:
    prompt = build_prompt(query, sources)
    raw = raw_generation(tokenizer, model, prompt)
    return parse_answer(raw)
