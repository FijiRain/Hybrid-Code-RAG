DEF_MODEL = "Qwen/Qwen3-0.6B"
SEM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEF_MAX_CHUNK_SIZE = 2000


SOURCES_PATH = "data/raw/vllm-0.10.1"
DATA_RAW = "data/raw/vllm-0.10.1"
DATA_CHUNKS = "data/processed/chunking_output"
DATA_INDEX = "data/processed/BM25_index"
SEM_CHUNKS = "data/processed/Semantic_index"
SAVE_SEARCH = "data/output/search_results/"
SEARCH_CODE = ("data/output/search_results/"
               "dataset_code_public.json")
SEARCH_DOCS = ("data/output/search_results/"
               "dataset_docs_public.json")
DATA_DOCS = ("datasets_public/public/UnansweredQuestions/"
             "dataset_docs_public.json")
DATA_CODE = ("datasets_public/public/UnansweredQuestions/"
             "dataset_code_public.json")
ANSWER_DOCS = ("datasets_public/public/AnsweredQuestions/"
               "dataset_docs_public.json")
ANSWER_CODE = ("datasets_public/public/AnsweredQuestions/"
               "dataset_code_public.json")
SEARCH_AND_ANSWER = "data/output/search_results_and_answer/"
