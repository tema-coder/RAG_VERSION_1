from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SUPPORTED_EXTENSIONS = [
    ".pdf",
    ".xlsx",
    ".xls",
    ".csv",
    ".docx",
    ".html",
    ".htm",
    ".txt",
    ".md",
]


class Settings(BaseSettings):
    MAX_FILE_SIZE_MB: int = 50
    DEFAULT_CHUNK_SIZE: int = 200
    DEFAULT_OVERLAP: int = 50
    DEFAULT_TABLE_CHUNK_SIZE: int = 5
    SEMANTIC_MODEL_NAME: str = "deepvk/USER-bge-m3"
    SEMANTIC_THRESHOLD: float = 0.5
    TEMP_DIR: str = "./temp"
    LOG_LEVEL: str = "INFO"
    DOC_STORE_MAX_ITEMS: int = 64
    SUPPORTED_EXTENSIONS: List[str] = DEFAULT_SUPPORTED_EXTENSIONS

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    @field_validator("TEMP_DIR")
    @classmethod
    def _expand_temp_dir(cls, value: str) -> str:
        return str(Path(value).expanduser())

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def supported_extensions_set(self) -> set:
        return {ext.lower() for ext in self.SUPPORTED_EXTENSIONS}

    def ensure_temp_dir(self) -> Path:
        path = Path(self.TEMP_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
