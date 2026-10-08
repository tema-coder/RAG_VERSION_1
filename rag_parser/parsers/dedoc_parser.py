"""Единый парсер документов через ``dedoc`` (9 форматов).

``DedocManager.parse(file_path, parameters={...})`` сам выбирает ридер по
расширению. Используется урезанный ``manager_config`` (только нужные
ридеры/конвертеры, PDF — только текстовый слой): дефолтная конфигурация
поднимает все ридеры dedoc, включая ``PdfImageReader`` (torchvision, OCR),
что запрещено ограничениями (только CPU, без OCR).

Артефакты на диске (``output/dedoc_results/``):

* ``raw_dedoc_tree_{document_id}.json`` — сырой JSON dedoc для отладки;
* ``parsed_{document_id}.md`` — Markdown-рендер полного дерева.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import orjson

from rag_parser.core.interfaces import (
    BaseParser,
    DocumentParseError,
    ParseResult,
    UnsupportedFormatError,
)

SUPPORTED_EXTENSIONS: List[str] = [
    ".docx",
    ".xlsx",
    ".xls",
    ".pdf",
    ".html",
    ".htm",
    ".txt",
    ".csv",
    ".tsv",
]

TREE_FILE_PREFIX = "raw_dedoc_tree"
MARKDOWN_FILE_PREFIX = "parsed"

_manager: Any = None


def build_manager() -> Any:
    """DedocManager с урезанным конфигом (без OCR/GPU-ридеров).

    Ридеры: DOCX, Excel, TXT, CSV/TSV, HTML, PDF только с текстовым слоем
    (``PdfTxtlayerReader``/``PdfTabbyReader``/``PdfBrokenEncodingReader`` —
    без ``PdfAutoReader``/``PdfImageReader``, тянущих torchvision и OCR).
    """
    from dedoc import DedocManager
    from dedoc.attachments_handler import AttachmentsHandler
    from dedoc.config import get_config
    from dedoc.converters import ConverterComposition
    from dedoc.converters.concrete_converters.docx_converter import DocxConverter
    from dedoc.converters.concrete_converters.excel_converter import ExcelConverter
    from dedoc.converters.concrete_converters.pdf_converter import PDFConverter
    from dedoc.converters.concrete_converters.txt_converter import TxtConverter
    from dedoc.metadata_extractors import MetadataExtractorComposition
    from dedoc.metadata_extractors.concrete_metadata_extractors import (
        base_metadata_extractor,
        docx_metadata_extractor,
        pdf_metadata_extractor,
    )
    from dedoc.readers import ReaderComposition
    from dedoc.readers.csv_reader.csv_reader import CSVReader
    from dedoc.readers.docx_reader.docx_reader import DocxReader
    from dedoc.readers.excel_reader.excel_reader import ExcelReader
    from dedoc.readers.html_reader.html_reader import HtmlReader
    from dedoc.readers.pdf_reader.pdf_txtlayer_reader.pdf_broken_encoding_reader import (
        PdfBrokenEncodingReader,
    )
    from dedoc.readers.pdf_reader.pdf_txtlayer_reader.pdf_tabby_reader import (
        PdfTabbyReader,
    )
    from dedoc.readers.pdf_reader.pdf_txtlayer_reader.pdf_txtlayer_reader import (
        PdfTxtlayerReader,
    )
    from dedoc.readers.txt_reader.raw_text_reader import RawTextReader
    from dedoc.structure_constructors import StructureConstructorComposition
    from dedoc.structure_constructors.concrete_structure_constructors import (
        tree_constructor,
    )
    from dedoc.structure_extractors import StructureExtractorComposition
    from dedoc.structure_extractors.concrete_structure_extractors import (
        default_structure_extractor,
    )

    config = get_config()
    structure_extractor_cls = default_structure_extractor.DefaultStructureExtractor
    tree_constructor_instance = tree_constructor.TreeConstructor()
    manager = DedocManager(
        config=config,
        manager_config={
            "converter": ConverterComposition(
                converters=[
                    DocxConverter(config=config),
                    ExcelConverter(config=config),
                    TxtConverter(config=config),
                    PDFConverter(config=config),
                ]
            ),
            "reader": ReaderComposition(
                readers=[
                    DocxReader(config=config),
                    ExcelReader(config=config),
                    RawTextReader(config=config),
                    CSVReader(config=config),
                    HtmlReader(config=config),
                    PdfTxtlayerReader(config=config),
                    PdfTabbyReader(config=config),
                    PdfBrokenEncodingReader(config=config),
                ]
            ),
            "structure_extractor": StructureExtractorComposition(
                extractors={
                    structure_extractor_cls.document_type: structure_extractor_cls(
                        config=config
                    )
                },
                default_key=structure_extractor_cls.document_type,
                config=config,
            ),
            "structure_constructor": StructureConstructorComposition(
                constructors={
                    "tree": tree_constructor_instance,
                    "": tree_constructor_instance,
                },
                default_constructor=tree_constructor_instance,
            ),
            "document_metadata_extractor": MetadataExtractorComposition(
                extractors=[
                    docx_metadata_extractor.DocxMetadataExtractor(config=config),
                    pdf_metadata_extractor.PdfMetadataExtractor(config=config),
                    base_metadata_extractor.BaseMetadataExtractor(config=config),
                ]
            ),
            "attachments_handler": AttachmentsHandler(config=config),
        },
    )
    return manager


def get_manager() -> Any:
    """Кэшированный ``DedocManager`` (сборка импортирует половину dedoc)."""
    global _manager
    if _manager is None:
        _manager = build_manager()
    return _manager


def dedoc_version() -> str:
    try:
        import dedoc.version

        return dedoc.version.__version__
    except Exception:  # noqa: BLE001 - version info is optional metadata
        return "unknown"


def detect_text_encoding(file_path: Path) -> Optional[str]:
    """Определяет кодировку текстового файла: строгий UTF-8, иначе None.

    Детектор dedoc (charset_normalizer) на коротких кириллических файлах
    ошибается (UTF-8 → utf_16_be, кракозябры), поэтому явный UTF-8
    передаётся параметром ``encoding`` для csv/tsv.
    """
    try:
        raw = file_path.read_bytes()[:65536]
    except OSError:
        return None
    if not raw:
        return None
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"


def parameters_for_format(extension: str, file_path: Path) -> Dict[str, str]:
    """Параметры ``DedocManager.parse`` для формата (таблица в README)."""
    params: Dict[str, str] = {
        "with_attachments": "false",
        "structure_type": "tree",
    }
    if extension == ".pdf":
        params["pdf_with_text_layer"] = "true"
    if extension in (".csv", ".tsv"):
        from rag_parser.parsers.base import detect_dialect

        try:
            sample = file_path.read_bytes()[:8192].decode("utf-8", errors="ignore")
        except OSError:
            sample = ""
        if extension == ".tsv":
            params["delimiter"] = "\t"
        elif sample:
            params["delimiter"] = detect_dialect(sample)
        encoding = detect_text_encoding(file_path)
        if encoding is not None:
            params["encoding"] = encoding
    return params


def convert_xls_to_xlsx(source: Path) -> Path:
    """Конвертирует legacy `.xls` (BIFF) во временный `.xlsx`.

    dedoc обрабатывает `.xls` только через LibreOffice (`soffice`), который
    исключён ограничениями, поэтому конвертация выполняется здесь:
    чтение через ``xlrd``, запись через ``openpyxl``. Только CPU.
    Объединённые диапазоны (``sheet.merged_cells``, есть в BIFF8; в старых
    BIFF5 список может быть пуст — см. README) переносятся как есть, чтобы
    точная заливка по диапазонам работала и для xls.
    """
    import tempfile

    import xlrd
    from openpyxl import Workbook

    try:
        try:
            # formatting_info нужен для sheet.merged_cells (BIFF8);
            # на старых BIFF5 падает — тогда мёрджы недоступны.
            book = xlrd.open_workbook(str(source), formatting_info=True)
        except Exception:  # noqa: BLE001 - старые BIFF без formatting
            book = xlrd.open_workbook(str(source))
    except Exception as exc:
        raise DocumentParseError(
            f"Failed to read legacy .xls workbook: {exc}", filename=source.name
        ) from exc
    workbook = Workbook()
    first = True
    for sheet in book.sheets():
        target = workbook.active if first else workbook.create_sheet()
        first = False
        target.title = sheet.name[:31] or "Sheet"
        for row_idx in range(sheet.nrows):
            target.append([sheet.cell_value(row_idx, col) for col in range(sheet.ncols)])
        for rlo, rhi, clo, chi in sheet.merged_cells:
            target.merge_cells(
                start_row=rlo + 1,
                start_column=clo + 1,
                end_row=rhi,
                end_column=chi,
            )
    handle = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    handle.close()
    target_path = Path(handle.name)
    workbook.save(target_path)
    return target_path


_CONTAINER_SIGNATURES = {
    ".docx": (b"PK\x03\x04", "ZIP (OOXML)"),
    ".xlsx": (b"PK\x03\x04", "ZIP (OOXML)"),
    ".pdf": (b"%PDF", "PDF"),
}


class DedocParser(BaseParser):
    """Единственный парсер сервиса: dedoc → дерево → Markdown."""

    format = "dedoc"
    extensions = tuple(SUPPORTED_EXTENSIONS)

    def __init__(self, header_threshold: Optional[float] = None) -> None:
        self.header_threshold = header_threshold

    def supported_extensions(self) -> List[str]:
        return list(SUPPORTED_EXTENSIONS)

    def parse(self, file_path: Path) -> ParseResult:
        from rag_parser.parsers.dedoc_tree import (
            build_full_tree,
            full_tree_to_markdown,
            reattach_tables,
        )
        from rag_parser.parsers.header_promoter import promote_headers
        from rag_parser.parsers.merged_cells import fill_workbook_tables

        path = Path(file_path)
        if not path.exists():
            raise DocumentParseError(f"File not found: {path}", filename=path.name)
        try:
            size_bytes = path.stat().st_size
        except OSError as exc:
            raise DocumentParseError(
                f"Cannot read file: {exc}", filename=path.name
            ) from exc
        if size_bytes == 0:
            raise DocumentParseError("File is empty", filename=path.name)

        extension = path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(extension, SUPPORTED_EXTENSIONS)

        self._validate_container(path, extension)

        # Legacy .xls dedoc умеет только через LibreOffice — конвертируем сами.
        converted: Optional[Path] = None
        effective_path = path
        effective_extension = extension
        if extension == ".xls":
            converted = convert_xls_to_xlsx(path)
            effective_path = converted
            effective_extension = ".xlsx"

        try:
            manager = get_manager()
        except ImportError as exc:
            raise DocumentParseError(
                f"dedoc is not installed: {exc}", filename=path.name
            ) from exc

        try:
            parsed = manager.parse(
                str(effective_path),
                parameters=parameters_for_format(effective_extension, effective_path),
            )
            document = parsed.to_api_schema()
            tree = document.model_dump()
        except (DocumentParseError, UnsupportedFormatError):
            if converted is not None:
                converted.unlink(missing_ok=True)
            raise
        except Exception as exc:  # noqa: BLE001 - dedoc raises assorted errors
            if converted is not None:
                converted.unlink(missing_ok=True)
            raise DocumentParseError(
                f"dedoc failed to parse the document: {exc}", filename=path.name
            ) from exc

        content = tree.get("content", {}) or {}
        tables = content.get("tables") or []
        if extension in (".xlsx", ".xls"):
            # Точная заливка по диапазонам книги (для xls — по
            # сконвертированному файлу с перенесёнными мёрджами);
            # файл нужен до конца заливки, удаляем после.
            try:
                fill_workbook_tables(tables, effective_path)
            finally:
                if converted is not None:
                    converted.unlink(missing_ok=True)
        elif converted is not None:
            converted.unlink(missing_ok=True)
        structure = content.get("structure") or {}

        full_root = build_full_tree(structure, None)
        full_root = promote_headers(full_root, threshold=self.header_threshold)
        # Привязка таблиц СТРОГО после промоции: якорь с таблицей-ребёнком
        # шёл бы composite-путём без резки строк (см. reattach_tables).
        reattach_tables(full_root, tables)
        markdown = full_tree_to_markdown(full_root)

        if not markdown.strip():
            if extension == ".pdf":
                raise DocumentParseError(
                    "PDF без текстового слоя не поддерживается, OCR отключён",
                    filename=path.name,
                )
            raise DocumentParseError(
                "Converted document contains no text", filename=path.name
            )
        markdown = self._clean(markdown)

        document_id = self._new_document_id()
        paths = self.save_results(document_id, tree, markdown)
        # В ParseResult.tree — ПОЛНОЕ дерево (чанкинг работает по полным
        # текстам). Урезание до превью делает слой API/UI последним шагом.
        return ParseResult(
            document_id=document_id,
            filename=path.name,
            format=self.format,
            markdown=markdown,
            metadata={
                "format": self.format,
                "extension": extension,
                "size_bytes": size_bytes,
                "lines": len(markdown.splitlines()),
                "characters": len(markdown),
                "parser": "dedoc",
                "dedoc_version": dedoc_version(),
                "tables": len(tables),
                "raw_tree_path": str(paths["raw_tree"]),
                "markdown_path": str(paths["markdown"]),
                "output_dir": str(paths["output_dir"]),
            },
            tree=full_root,
        )

    def _validate_container(self, path: Path, extension: str) -> None:
        """Отклоняет файлы, чьи байты не соответствуют расширению.

        Без проверки dedoc молча пропускает битый контейнер через
        fallback-конвертеры и возвращает мусор вместо ошибки.
        """
        expected = _CONTAINER_SIGNATURES.get(extension)
        if expected is None:
            return
        signature, description = expected
        try:
            with path.open("rb") as handle:
                header = handle.read(len(signature))
        except OSError as exc:
            raise DocumentParseError(
                f"Cannot read file: {exc}", filename=path.name
            ) from exc
        if header != signature:
            raise DocumentParseError(
                f"File content does not match the {extension} extension "
                f"(expected a {description} container)",
                filename=path.name,
            )

    def _clean(self, text: str) -> str:
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n\n")
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def save_results(
        self, document_id: str, tree: Dict[str, Any], markdown: str
    ) -> Dict[str, Path]:
        """Пишет сырой JSON dedoc и Markdown-рендер на диск."""
        from rag_parser.core.config import settings

        output_dir = settings.ensure_dedoc_output_dir()
        tree_path = output_dir / f"{TREE_FILE_PREFIX}_{document_id}.json"
        markdown_path = output_dir / f"{MARKDOWN_FILE_PREFIX}_{document_id}.md"
        tree_path.write_bytes(
            orjson.dumps(tree, option=orjson.OPT_INDENT_2) + b"\n"
        )
        markdown_path.write_bytes(markdown.encode("utf-8"))
        return {
            "output_dir": output_dir,
            "raw_tree": tree_path,
            "markdown": markdown_path,
        }


__all__ = [
    "DedocParser",
    "SUPPORTED_EXTENSIONS",
    "MARKDOWN_FILE_PREFIX",
    "TREE_FILE_PREFIX",
    "convert_xls_to_xlsx",
    "dedoc_version",
    "detect_text_encoding",
    "get_manager",
    "parameters_for_format",
]
