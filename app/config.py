"""Configuration settings for Veridict application."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API Keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/veridict"

    # Application
    app_env: str = "development"
    debug: bool = True
    app_name: str = "Veridict"
    app_version: str = "0.1.0"

    # Processing
    max_workers: int = 10
    voting_k_margin: int = 3

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.app_env == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
