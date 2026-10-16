from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DEDOC_OUTPUT_DIR = str(PACKAGE_ROOT / "output" / "dedoc_results")

DEFAULT_SUPPORTED_EXTENSIONS = [
    # Офисные (dedoc конвертирует .doc/.odt/.rtf через LibreOffice)
    ".docx",
    ".doc",
    ".odt",
    ".rtf",
    # Табличные
    ".xlsx",
    ".xls",
    ".csv",
    ".tsv",
    # PDF (только текстовый слой, OCR отключён)
    ".pdf",
    # Текстовые и разметка
    ".html",
    ".htm",
    ".txt",
    ".xml",
    ".md",
]


class Settings(BaseSettings):
    MAX_FILE_SIZE_MB: int = 50
    DEFAULT_CHUNK_SIZE: int = 200
    DEFAULT_OVERLAP: int = 50
    SEMANTIC_MODEL_NAME: str = "deepvk/USER-bge-m3"
    SEMANTIC_THRESHOLD: float = 0.5
    TEMP_DIR: str = "./temp"
    DEDOC_OUTPUT_DIR: str = DEFAULT_DEDOC_OUTPUT_DIR
    LOG_LEVEL: str = "INFO"
    DOC_STORE_MAX_ITEMS: int = 64
    SUPPORTED_EXTENSIONS: List[str] = DEFAULT_SUPPORTED_EXTENSIONS
    # ТЗ 3.2: в крошку входят максимум ДВА последних уровня заголовков.
    # Например 1.1 -> 1.1.1 -> 1.1.1.1 даёт "1.1.1 Название > 1.1.1.1 Название".
    # 0 или меньше отключает ограничение (весь путь) — для отладки.
    BREADCRUMB_MAX_DEPTH: int = 2
    TREE_TEXT_PREVIEW_LEN: int = 30
    # PDF → DOCX через pdf2docx перед разбором. ВКЛЮЧЕНО по умолчанию:
    # нативный PdfTxtlayerReader на gost_report.pdf даёт 218 «заголовков»
    # (числа в ячейках таблиц, повышенные header_promoter) и 0 таблиц,
    # тогда как маршрут через pdf2docx даёт 37 заголовков и 9 таблиц —
    # структура документа сохраняется, см. README, «Ограничения PDF».
    # Цена: конвертер склеивает слова без пробелов и теряет часть титула.
    PDF_VIA_DOCX: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    @field_validator("TEMP_DIR", "DEDOC_OUTPUT_DIR")
    @classmethod
    def _expand_dir(cls, value: str) -> str:
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

    def ensure_dedoc_output_dir(self) -> Path:
        path = Path(self.DEDOC_OUTPUT_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
