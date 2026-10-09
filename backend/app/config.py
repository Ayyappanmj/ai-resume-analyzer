"""Central configuration (env vars / .env)."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Optional. Example: postgresql+psycopg2://resume:resume@localhost:5432/resume
    database_url: str | None = None

    max_upload_mb: int = 5
    cors_origins: str = "*"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
