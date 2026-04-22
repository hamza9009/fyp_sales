"""
Application configuration using pydantic-settings.
All values are read from environment variables or .env file.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/ecommerce_analytics"

    # Application
    APP_NAME: str = "Predictive Analytics SaaS"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # API
    API_PREFIX: str = "/api/v1"

    # ML artifacts
    MODEL_ARTIFACTS_DIR: str = "./ml/artifacts"

    # Processed data (Phase 2 ETL output)
    PROCESSED_DATA_DIR: str = "./data/processed"

    # Forecast horizon (days)
    DEFAULT_FORECAST_HORIZON: int = 7

    # Inventory simulation — lead time in days (assumption; UCI has no stock data)
    INVENTORY_LEAD_TIME_DAYS: int = 7
    # Starting stock multiplier: initial stock = multiplier × avg_daily_demand
    INVENTORY_STOCK_MULTIPLIER: float = 30.0
    INVENTORY_SIM_HISTORY_DAYS: int = 120
    INVENTORY_TARGET_COVER_DAYS: float = 35.0
    INVENTORY_SAFETY_STOCK_DAYS: float = 3.0
    INVENTORY_MONTE_CARLO_RUNS: int = 200

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
