import os
from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "hf_vector_db")
DOCS_TEMP_DIR = os.path.join(DATA_DIR, "temp_docs")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "stepfun/step-3.5-flash:free"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

LIBS_CONFIG = {
    "transformers": ("https://github.com/huggingface/transformers.git", "docs/source/en"),
    "peft": ("https://github.com/huggingface/peft.git", "docs/source"),
    "accelerate": ("https://github.com/huggingface/accelerate.git", "docs/source"),
    "trl": ("https://github.com/huggingface/trl.git", "docs/source"),
    "datasets": ("https://github.com/huggingface/datasets.git", "docs/source"),
}
