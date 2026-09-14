from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    ollama_timeout_seconds: int = Field(default=120, ge=1)

    raw_data_directory: str = "data/raw"
    processed_data_directory: str = "data/processed"


@lru_cache
def get_settings() -> Settings:
    return Settings()
