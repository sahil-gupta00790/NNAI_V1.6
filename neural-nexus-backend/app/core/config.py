# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict # Updated import for Pydantic v2+
# --- NEW: Import RedisDsn ---
from pydantic import RedisDsn
# --- End Import ---
import os
from dotenv import load_dotenv

# Load .env file from the project root (adjust path if needed)
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(dotenv_path=dotenv_path)
# print(f"Loading .env from: {dotenv_path}") # Debug print

class Settings(BaseSettings):
    # Paths
    UPLOAD_DIR: str = "/code/uploads"
    RESULT_DIR: str = "/code/results"
    STANDARD_EVAL_SCRIPT_PATH: str = "/code/app/standard_eval/mnist_eval.py"
    RAG_STORAGE_DIR: str = "/code/storage"
    RAG_DATA_DIR: str = "/code/data/research_papers"

    # API Keys & Settings
    HF_TOKEN: str | None = None
    GEMINI_API_KEY: str | None = None
    ALLOWED_ORIGINS: str = "http://localhost:3000"
    GEMINI_MODEL_NAME: str = "gemini-1.5-flash" # Added based on analysis.py usage

    # Celery
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"

    # --- NEW: Redis URL for Halt Flags (Using RedisDsn) ---
    # Default points to 'redis' service name on DB 1 (separate from Celery)
    # Can be overridden by REDIS_URL in .env file
    REDIS_URL: RedisDsn = "redis://redis:6379/1"
    # --- End Redis URL ---

    # RAG Models (Defaults match qa_system.py, can be overridden by .env)
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-base-en-v1.5"
    # NOTE: Using 1B parameter model for compatibility with lower VRAM
    LLM_MODEL_NAME: str = "meta-llama/Llama-3.1-8B-Instruct" # Default from original, ensure compatibility
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"
    ENABLE_RAG: bool = True # Added based on main.py usage

    # --- Security ---
    # Secret key for potential future use (e.g., JWT)
    # SECRET_KEY: str = "generate_a_strong_secret_key"

    # Updated Config for Pydantic v2+
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        case_sensitive=False # Environment variables are usually uppercase
    )
    # --- Deprecated Pydantic v1 Config ---
    # class Config:
    #     env_file = '.env'
    #     env_file_encoding = 'utf-8'
    #     case_sensitive = False

settings = Settings()

# --- Validate Paths (Optional) ---
# (Uncomment if needed)
# logger = logging.getLogger(__name__) # Add import logging if using logger here
# try:
#     for path in [settings.UPLOAD_DIR, settings.RESULT_DIR, settings.RAG_STORAGE_DIR]:
#         os.makedirs(path, exist_ok=True)
#     logger.info("Checked/created necessary directories.")
# except OSError as e:
#     logger.error(f"Error creating directories: {e}")

# --- Log Loaded Settings (Optional Debugging) ---
# print("--- Loaded Configuration ---")
# print(f"UPLOAD_DIR: {settings.UPLOAD_DIR}")
# print(f"RESULT_DIR: {settings.RESULT_DIR}")
# print(f"RAG_STORAGE_DIR: {settings.RAG_STORAGE_DIR}")
# print(f"REDIS_URL: {settings.REDIS_URL}") # Log the new setting
# print(f"HF_TOKEN Provided: {'Yes' if settings.HF_TOKEN else 'No'}")
# print(f"LLM Model: {settings.LLM_MODEL_NAME}")
# print("--------------------------")

