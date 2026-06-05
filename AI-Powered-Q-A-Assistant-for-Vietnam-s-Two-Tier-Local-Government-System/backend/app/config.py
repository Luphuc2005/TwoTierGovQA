"""
Application configuration — loaded from environment variables / .env file.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── Database ──────────────────────────────────────────────
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5433/legal_chatbot"

    # ── JWT ───────────────────────────────────────────────────
    JWT_SECRET_KEY: str = "change-me-to-a-very-long-random-secret-string"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    # ── RAG Pipeline ──────────────────────────────────────────
    RAG_BACKEND: str = "gemini"
    RAG_MODEL: str = "gemini-2.5-flash"
    RAG_MAX_TOKENS: int = 2048
    RAG_TEMPERATURE: float = 0.1
    GEMINI_API_KEY: str = ""

    # ── Redis ─────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6380/0"

    # ── Application ───────────────────────────────────────────
    APP_ENV: str = "development"
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"
    RATE_LIMIT_PER_MINUTE: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
