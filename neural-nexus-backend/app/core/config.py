# app/core/config.py
from pydantic_settings import BaseSettings
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
    ALLOWED_ORIGINS: str = "http://localhost:3000"

    # Celery
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"

    # RAG Models (Defaults match qa_system.py, can be overridden by .env)
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-base-en-v1.5"
    LLM_MODEL_NAME: str = "meta-llama/Llama-3.2-1B-Instruct" # Use a smaller model for 4GB VRAM
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"

    # --- Security ---
    # Secret key for potential future use (e.g., JWT)
    # SECRET_KEY: str = "generate_a_strong_secret_key"

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False # Environment variables are usually uppercase

settings = Settings()

# --- Validate Paths ---
# You might want basic checks here, although Docker volumes handle creation
# if not os.path.exists(settings.UPLOAD_DIR): os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
# if not os.path.exists(settings.RESULT_DIR): os.makedirs(settings.RESULT_DIR, exist_ok=True)

# --- Log Loaded Settings (Optional Debugging) ---
# print("--- Loaded Configuration ---")
# print(f"UPLOAD_DIR: {settings.UPLOAD_DIR}")
# print(f"RESULT_DIR: {settings.RESULT_DIR}")
# print(f"RAG_STORAGE_DIR: {settings.RAG_STORAGE_DIR}")
# print(f"HF_TOKEN Provided: {'Yes' if settings.HF_TOKEN else 'No'}")
# print(f"LLM Model: {settings.LLM_MODEL_NAME}")
# print("--------------------------")
