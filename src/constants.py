DEF_MODEL = "Qwen/Qwen3-0.6B"
DEF_MAX_CHUNK_SIZE = 2000


SOURCES_PATH = "data/raw/vllm-0.10.1"
DATA_RAW = "data/raw/vllm-0.10.1"
DATA_CHUNKS = "data/processed/chunking_output"
DATA_INDEX = "data/processed/BM25_index"
SAVE_SEARCH = "data/output/search_results/UnansweredQuestions"
SEARCH_CODE = ("data/output/search_results/UnansweredQuestions/"
               "dataset_code_public.json")
SEARCH_DOCS = ("data/output/search_results/UnansweredQuestions/"
               "dataset_docs_public.json")
DATA_DOCS = ("datasets_public/public/UnansweredQuestions/"
             "dataset_docs_public.json")
DATA_CODE = ("datasets_public/public/UnansweredQuestions/"
             "dataset_code_public.json")
ANSWER_DOCS = ("datasets_public/public/AnsweredQuestions/"
               "dataset_docs_public.json")
ANSWER_CODE = ("datasets_public/public/AnsweredQuestions/"
               "dataset_code_public.json")
