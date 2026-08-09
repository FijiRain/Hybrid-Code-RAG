import bm25s
from rich import print
from pathlib import Path

from .constants import DEF_MAX_CHUNK_SIZE, DATA_INDEX
from .data_models import RagDataset, MinimalSearchResults, StudentSearchResults
from .indexing import build_index
from .searching import search_sources
from .answering import generate_simple_answer, load_model
from .utils import reload_chunks, save_searchresults

'''
copypasta

index           : uv run python -m src index
search          : uv run python -m src search
                  "How to configure the OpenAI server?" --k 5
search dataset  : uv run python -m src search_dataset --dataset_path datasets_public/public/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
moulinette      : ./moulinette evaluate_student_search_results
                  output/search_results/UnansweredQuestions/
                  UnansweredQuestions.json
                  datasets_public/public/AnsweredQuestions/
                  dataset_code_public.json --k 10
                  --max_context_length 2000

answer_dataset  : uv run python -m src answer_dataset
                  --student_search_results_path
                  data/output/search_results/UnansweredQuestions/
                  dataset_docs_public.json
                  --save_directory
                  data/output/search_results_and_answer/
                  UnansweredQuestions
'''


class Cli():
    def index(self, max_chunk_size: int = DEF_MAX_CHUNK_SIZE) -> None:
        build_index(max_chunk_size)

    def search(self, query: str, k: int) -> None:
        if not query:
            raise ValueError("Query cannnot be empty.")
        if k <= 0:
            raise ValueError("k must be > 0.")

        all_chunks = reload_chunks()
        retriever = bm25s.BM25.load(DATA_INDEX)
        results = search_sources(query, k, all_chunks, retriever)
        for source in results:
            print(
                f"{source.file_path} "
                f"[{source.first_character_index}:"
                f"{source.last_character_index}]"
            )

    def search_dataset(self, dataset_path: str, k: int,
                       save_directory: str) -> None:
        try:
            with open(dataset_path, "r") as f:
                dataset = RagDataset.model_validate_json(f.read())
        except FileNotFoundError:
            raise FileNotFoundError("Incorrect dataset_path value.")
        if k <= 0:
            raise ValueError("k must be > 0.")

        all_chunks = reload_chunks()
        retriever = bm25s.BM25.load(DATA_INDEX)
        minsearch_list = []
        for uq in dataset.rag_questions:
            top_sources = search_sources(uq.question, k, all_chunks, retriever)
            ms_result = MinimalSearchResults(question_id=uq.question_id,
                                             question=uq.question,
                                             retrieved_sources=top_sources)
            minsearch_list.append(ms_result)

        result = StudentSearchResults(search_results=minsearch_list, k=k)
        filename = Path(dataset_path).name
        save_searchresults(result, save_directory, filename)
        print(f"[bold green]Saved student_search_results "
              f"to {save_directory}/{filename}[/bold green]")

    def answer(self, query: str, k: int) -> None:
        if not query:
            raise ValueError("Query cannnot be empty.")
        if k <= 0:
            raise ValueError("k must be > 0.")

        all_chunks = reload_chunks()
        retriever = bm25s.BM25.load(DATA_INDEX)
        search_res = search_sources(query, k, all_chunks, retriever)
        tokenizer, model = load_model()
        answer_text = generate_simple_answer(tokenizer, model,
                                             query, search_res)
        print(answer_text)

    def diagnose_thinking_length(self):
        from .answering import build_prompt, raw_generation
        import time
        tokenizer, model = load_model()

        # test_questions = [
        #     "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?",
        #     "What command can be used to evaluate the accuracy of a quantized model using lm_eval with vLLM?",
        #     "What method does vLLM's LLM class provide for generating embedding vectors from prompts?",
        #     "What hardware platforms does vLLM support?",
        #     "What are the differences between mm_kwargs and tok_kwargs when using the _call_hf_processor method in vLLM multimodal processing?"
        #     "Where can I find information about using generative models in vLLM?",
        #     "How is the number of placeholder feature tokens for an image calculated in vLLM's multimodal implementation?",
        #     "What parallelism strategy does vLLM support for large-scale deployment of Mixture of Experts models?",
        #     "How do you achieve reproducible results in vLLM?",
        #     "Which vLLM model architectures support LoRA fine-tuning?"
        # ]
        test_questions = [
            # "What activation formats does the fused batched MoE layer return in vLLM?",
            # "What are the default values for FP8_MIN and FP8_MAX constants in vLLM's triton_flash_attention module?",
            # "What determines whether vLLM's sampler returns Pythonized results or deferred Pythonization arguments?",
            # "What's the default value of trust_remote_code in vLLM's LLM class constructor?",
            # "What determines the values in cudagraph_inputs_embeds when capturing CUDA graph shapes in vLLM's ModelRunner?",
            "What conditions must be met for vLLM's ModelRunner to use CUDA graphs instead of the regular model?",
            # "What is the default timeout value for vLLM RPC operations?",
            # "What does the Gemma3ForCausalLM constructor assert about the tie_word_embeddings configuration?",
            "What value is passed for ngroups when has_groups is False in the _bmm_chunk_fwd_kernel call?",
            # "What is the default cudagraph_support value for TritonAttentionMetadataBuilder?" 
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

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        ...

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        ...
