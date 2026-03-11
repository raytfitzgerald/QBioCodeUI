import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_TITLE: str = "QBioCode UI"
    APP_VERSION: str = "0.1.0"

    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DATASETS_DIR: Path = DATA_DIR / "datasets"
    RESULTS_DIR: Path = DATA_DIR / "results"

    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB

    DEFAULT_BACKEND: str = "simulator"
    DEFAULT_SEED: int = 42
    DEFAULT_SHOTS: int = 1024
    DEFAULT_RESIL_LEVEL: int = 1

    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    class Config:
        env_prefix = "QBIOCODE_"


settings = Settings()

# Ensure data directories exist
settings.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
