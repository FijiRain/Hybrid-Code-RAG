import json
from typing import Any
from pathlib import Path

from .constants import DATA_CHUNKS, DATA_INDEX
from .data_models import StudentSearchResults


def dump_chunks(py: list[dict[str, Any]], md: list[dict[str, Any]]) -> None:
    target_dir = Path(DATA_CHUNKS)
    target_dir.mkdir(parents=True, exist_ok=True)

    with (target_dir / "py_chunks.json").open("w") as f:
        json.dump(py, f, indent=2)

    with (target_dir / "md_chunks.json").open("w") as f:
        json.dump(md, f, indent=2)


def reload_chunks() -> list[dict[str, Any]]:
    target = Path(DATA_CHUNKS)
    try:
        with Path(target / "py_chunks.json").open("r") as f:
            py_chunks = json.load(f)

        with Path(target / "md_chunks.json").open("r") as f:
            md_chunks = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("You must index before searching anything.")

    chunks_reloaded: list[dict[str, Any]] = py_chunks + md_chunks
    return chunks_reloaded


def load_files(dir: str) -> list[tuple[str, str]]:
    res = []
    dir_path = Path(dir)
    extension = {".py", ".md"}

    selected_files = [str(f) for f in dir_path.rglob("*")
                      if f.suffix in extension]
    for file in selected_files:
        with open(file, "r") as content:
            res.append((file, content.read()))
    return res


def save_searchresults(search_res: StudentSearchResults,
                       save_dir: str,
                       filename: str) -> None:
    try:
        dir = Path(save_dir)
        dir.mkdir(parents=True, exist_ok=True)
        file_path = dir / filename
        file_path.write_text(search_res.model_dump_json(indent=2))
    except OSError:
        raise OSError("Incorrect save_directory value.")


def thinking_length():
    from .answering import build_prompt, raw_generation, load_model
    from .indexing import bm25s
    from .searching import search_sources
    import time
    tokenizer, model = load_model()

    # Code questions
    test_questions = [
        (
            "What activation formats does the fused batched MoE layer return "
            "in vLLM?"
        ),
        (
            "What are the default values for FP8_MIN and FP8_MAX constants "
            "in vLLM's triton_flash_attention module?"
        ),
        (
            "What determines whether vLLM's sampler returns Pythonized "
            "results or deferred Pythonization arguments?"
        ),
        (
            "What's the default value of trust_remote_code in vLLM's LLM "
            "class constructor?"
        ),
        (
            "What determines the values in cudagraph_inputs_embeds when "
            "capturing CUDA graph shapes in vLLM's ModelRunner?"
        ),
        (
            "What conditions must be met for vLLM's ModelRunner to use CUDA "
            "graphs instead of the regular model?"
        ),
        "What is the default timeout value for vLLM RPC operations?",
        (
            "What does the Gemma3ForCausalLM constructor assert about the "
            "tie_word_embeddings configuration?"
        ),
        (
            "What value is passed for ngroups when has_groups is False in "
            "the _bmm_chunk_fwd_kernel call?"
        ),
        (
            "What is the default cudagraph_support value for "
            "TritonAttentionMetadataBuilder?"
        ),
    ]

    tot = 0
    for q in test_questions:
        all_chunks = reload_chunks()
        retriever = bm25s.BM25.load(DATA_INDEX)
        sources = search_sources(q, 5, all_chunks, retriever)

        prompt = build_prompt(q, sources)
        start = time.time()
        raw = raw_generation(tokenizer, model, prompt, 1024)
        elapsed = time.time() - start
        tot += elapsed
        print(f"{elapsed:.1f}s pour cette génération")

        n_tokens = len(tokenizer.encode(raw))
        closed = "</think>" in raw
        print(f"[{'OK' if closed else 'TRONQUÉ'}] {n_tokens} tokens — {q}")
    print(f"TOTAL FOR ALL GENERATIONS : {tot}")
