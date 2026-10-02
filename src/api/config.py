import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

# Automatically find the project root directory from src/api/config.py
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    QDRANT_URL: str = "http://localhost:6333"
    NEO4J_URI: str = "bolt://192.168.100.72:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")  # Satisfies Pylance; overwritten by .env at runtime
    VLLM_BASE_URL: str = "http://localhost:8000/v1"
    VLLM_MODEL_NAME: str = "casperhansen/mistral-nemo-instruct"

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings()