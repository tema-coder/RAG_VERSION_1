"""Единый фасад разбора документов: Docling (PDF/изображения) + dedoc.

``DedocManager.parse(file_path, parameters={...})`` сам выбирает ридер по
расширению. Используется урезанный ``manager_config`` (только нужные
ридеры/конвертеры, PDF — только текстовый слой): дефолтная конфигурация
поднимает все ридеры dedoc, включая ``PdfImageReader`` (torchvision, OCR),
что запрещено ограничениями (только CPU, без OCR).

Требования к форматам:

* ``.doc`` / ``.odt`` / ``.rtf`` — dedoc конвертирует их в DOCX вызовом
  ``soffice --headless``, поэтому в системе должен стоять LibreOffice
  (``sudo apt install libreoffice``). Без него эти форматы не читаются;
* ``.xml`` — свой препроцессор ``xml_preprocessor`` переписывает теги в HTML
  (иначе ``RawTextReader`` игнорирует их и структура теряется полностью);
* ``.md`` — ``markdown_preprocessor`` рендерит markdown в HTML через
  ``mistune`` (dedoc markdown не знает, а откат на ``text/plain`` —
  недокументированное поведение, и оно молча ломает разбор при смене версии);
* ``.pdf`` — обрабатывается пайплайном Docling (``_parse_via_docling``):
  ``DocumentConverter`` → ``export_to_markdown()`` → существующий
  ``.md``-маршрут (mistune → HtmlReader → дерево → чанкинг). dedoc больше
  не читает PDF напрямую;
* ``.png`` / ``.jpg`` / ``.jpeg`` — упаковываются в временный PDF через
  ``img2pdf`` и проходят тот же Docling-пайплайн;
* ``.xls`` — конвертируется в ``.xlsx`` через ``xlrd``+``openpyxl``,
  чтобы не зависеть от LibreOffice на legacy-файлах.

Артефакты на диске (``output/dedoc_results/``):

* ``raw_dedoc_tree_{document_id}.json`` — сырой JSON dedoc для отладки;
* ``parsed_{document_id}.md`` — Markdown-рендер полного дерева.
"""

import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

import orjson

from rag_parser.core.config import settings
from rag_parser.core.interfaces import (
    BaseParser,
    DocumentParseError,
    ParseResult,
    UnsupportedFormatError,
)

SUPPORTED_EXTENSIONS: List[str] = [
    ".docx",
    ".doc",
    ".odt",
    ".rtf",
    ".xlsx",
    ".xls",
    ".csv",
    ".tsv",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".html",
    ".htm",
    ".txt",
    ".xml",
    ".md",
]

#: Расширения изображений: упаковываются в PDF через img2pdf и идут в Docling.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

#: Расширения, которые dedoc не умеет читать нативно и которые мы
#: подменяем на ``.txt`` перед разбором (см. ``prepare_for_parsing``).
TEXT_ALIAS_EXTENSIONS: Dict[str, str] = {
    ".md": ".txt",
}

#: Расширения, требующие LibreOffice (конвертация в DOCX средствами dedoc).
LIBREOFFICE_EXTENSIONS = (".doc", ".odt", ".rtf")

#: Форматы, которые идут через HTML-маршрут (HtmlReader + маркеры таблиц).
#: ``.html`` — напрямую; остальные конвертируются в HTML предобработчиками
#: ``.md`` (mistune), ``.xml`` (xml_preprocessor), ``.txt`` (детектор таблиц).
HTML_DERIVED_EXTENSIONS = (".html", ".htm", ".md", ".xml", ".txt")

#: Подмножество ``HTML_DERIVED_EXTENSIONS``, где разметка источника сама
#: задаёт структуру (h1-h6, ``<section>``). Для них жёсткие правила
#: ``header_promoter`` НЕ применяются: dedoc уже расставил типы и уровни,
#: а правила подняли бы нумерованные пункты списка до заголовков.
#: ``.txt`` сюда НЕ входит — заголовки в нём выглядят как обычные строки,
#: их по-прежнему распознаёт ``header_promoter``.
MARKUP_AUTHORITATIVE_EXTENSIONS = (".html", ".htm", ".md", ".xml")

TREE_FILE_PREFIX = "raw_dedoc_tree"
MARKDOWN_FILE_PREFIX = "parsed"

_manager: Any = None


def build_manager() -> Any:
    """DedocManager с урезанным конфигом (без OCR/GPU-ридеров и без PDF).

    Ридеры: DOCX, Excel, TXT, CSV/TSV, HTML. PDF и изображения больше не
    читаются dedoc-ом напрямую — для них используется Docling
    (``_parse_via_docling``), а dedoc разбирает уже собранный Markdown
    через ``.md``-маршрут.
    """
    from dedoc import DedocManager
    from dedoc.attachments_handler import AttachmentsHandler
    from dedoc.config import get_config
    from dedoc.converters import ConverterComposition
    from dedoc.converters.concrete_converters.docx_converter import DocxConverter
    from dedoc.converters.concrete_converters.excel_converter import ExcelConverter
    from dedoc.converters.concrete_converters.txt_converter import TxtConverter
    from dedoc.metadata_extractors import MetadataExtractorComposition
    from dedoc.metadata_extractors.concrete_metadata_extractors import (
        base_metadata_extractor,
        docx_metadata_extractor,
    )
    from dedoc.readers import ReaderComposition
    from dedoc.readers.csv_reader.csv_reader import CSVReader
    from dedoc.readers.docx_reader.docx_reader import DocxReader
    from dedoc.readers.excel_reader.excel_reader import ExcelReader
    from dedoc.readers.html_reader.html_reader import HtmlReader
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
                ]
            ),
            "reader": ReaderComposition(
                readers=[
                    DocxReader(config=config),
                    ExcelReader(config=config),
                    RawTextReader(config=config),
                    CSVReader(config=config),
                    HtmlReader(config=config),
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


def prepare_manager_for_format(extension: str) -> Any:
    """Возвращает менеджер, готовый к разбору указанного формата.

    PDF и изображения больше не доходят до dedoc (обрабатываются Docling-
    пайплайном ``_parse_via_docling``), поэтому bbox-защита не нужна.
    """
    return get_manager()


def libreoffice_available() -> bool:
    """Есть ли ``soffice`` в PATH (нужен для .doc/.odt/.rtf)."""
    return (
        shutil.which("soffice") is not None or shutil.which("libreoffice") is not None
    )


def require_libreoffice(extension: str, filename: str) -> None:
    """Диагностика для .doc/.odt/.rtf без установленного LibreOffice.

    Без ``soffice`` dedoc не может сконвертировать файл в DOCX и падает
    невнятной ошибкой в глубине конвертера. Здесь сообщаем точную причину.
    """
    if extension.lower() not in LIBREOFFICE_EXTENSIONS:
        return
    if libreoffice_available():
        return
    raise DocumentParseError(
        f"Формат {extension} требует LibreOffice (soffice) для конвертации в "
        "DOCX. Установите его: sudo apt install libreoffice",
        filename=filename,
    )


class PreparedInput(NamedTuple):
    """Результат подготовки файла к разбору.

    :param path: путь, который нужно отдать dedoc;
    :param extension: расширение для выбора ридера;
    :param temp_path: временный файл, который обязан удалить вызывающий;
    :param html_tables: markdown-таблицы, извлечённые из HTML нами
        (для последующей привязки по маркерам, см.
        ``dedoc_tree.attach_tables_by_position``).
    """

    path: Path
    extension: str
    temp_path: Optional[Path]
    html_tables: List[str]


def prepare_for_parsing(path: Path) -> PreparedInput:
    """Готовит файл к разбору dedoc.

    Для HTML-производных (``HTML_DERIVED_EXTENSIONS``) таблицы
    извлекаются заранее и заменяются маркерами-абзацами — иначе
    ``HtmlReader`` не даёт ни позиции таблиц, ни ``TableAnnotation``
    (см. ``parsers.html_preprocessor``).
    """
    extension = path.suffix.lower()

    if extension in (".html", ".htm", ".md", ".xml", ".txt"):

        if extension == ".md":
            from rag_parser.parsers.markdown_preprocessor import (
                prepare_markdown_for_dedoc,
            )

            target_dir = settings.ensure_temp_dir()
            temp_path, tables = prepare_markdown_for_dedoc(path, target_dir)
            if temp_path == path:
                # файл не читается как текст — нечего конвертировать
                return PreparedInput(path, ".txt", None, [])
            return PreparedInput(temp_path, ".html", temp_path, tables)

        if extension == ".xml":
            from rag_parser.parsers.xml_preprocessor import prepare_xml_for_dedoc

            target_dir = settings.ensure_temp_dir()
            temp_path, tables = prepare_xml_for_dedoc(path, target_dir)
            if temp_path is None:
                # невалидный XML — не мешаем dedoc разобрать его самому
                return PreparedInput(path, ".txt", None, [])
            return PreparedInput(temp_path, ".html", temp_path, tables)

        if extension == ".txt":
            from rag_parser.parsers.text_preprocessor import prepare_text_for_dedoc

            target_dir = settings.ensure_temp_dir()
            temp_path, tables = prepare_text_for_dedoc(path, target_dir)
            if temp_path is None:
                # таблиц нет — нативный RawTextReader лучше сохраняет
                # переносы строк, чем наша HTML-обёртка
                return PreparedInput(path, ".txt", None, [])
            return PreparedInput(temp_path, ".html", temp_path, tables)

        from rag_parser.parsers.html_preprocessor import (
            inject_table_markers,
            promote_gost_structure,
        )

        try:
            html = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return PreparedInput(path, extension, None, [])
        marked, tables = inject_table_markers(promote_gost_structure(html))
        if not tables:
            return PreparedInput(path, extension, None, [])
        target_dir = settings.ensure_temp_dir()
        temp_path = target_dir / f"{path.stem}_marked{extension}"
        temp_path.write_text(marked, encoding="utf-8")
        return PreparedInput(temp_path, ".html", temp_path, tables)

    alias = TEXT_ALIAS_EXTENSIONS.get(extension)
    if alias is None:
        return PreparedInput(path, extension, None, [])


    target_dir = settings.ensure_temp_dir()
    alias_path = target_dir / f"{path.stem}{alias}"
    shutil.copyfile(path, alias_path)
    return PreparedInput(alias_path, alias, alias_path, [])


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
    if extension in HTML_DERIVED_EXTENSIONS:
        # КРИТИЧНО для HTML-маршрута: без этого флага dedoc ПОЛНОСТЬЮ
        # отбрасывает таблицы, у которых нет атрибута border= (а mistune
        # и наш xml/txt-препроцессоры его не проставляют). Таблица не
        # просто остаётся нераспознанной — она исчезает из вывода.
        params["handle_invisible_table"] = "true"
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
            target.append(
                [sheet.cell_value(row_idx, col) for col in range(sheet.ncols)]
            )
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
    ".png": (b"\x89PNG\r\n\x1a\n", "PNG"),
    ".jpg": (b"\xff\xd8\xff", "JPEG"),
    ".jpeg": (b"\xff\xd8\xff", "JPEG"),
    # Legacy OLE2-контейнер (OLE2 signature, D0CF11E0A1B11AE1)
    ".doc": (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "OLE2"),
    ".xls": (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "OLE2"),
    # ODF — ZIP-подобный контейнер, но с mimetype-членом вперемешку;
    # проверяем ZIP-заголовок, полная валидация — забота LibreOffice.
    ".odt": (b"PK\x03\x04", "ZIP (ODF)"),
}


class DedocParser(BaseParser):
    """Единственный парсер сервиса: dedoc → дерево → Markdown."""

    format = "dedoc"
    extensions = tuple(SUPPORTED_EXTENSIONS)

    def __init__(
        self,
        header_threshold: Optional[float] = None,  # noqa: ARG002 - DEPRECATED
    ) -> None:
        """Параметр header_threshold DEPRECATED и игнорируется."""

    def supported_extensions(self) -> List[str]:
        return list(SUPPORTED_EXTENSIONS)

    def parse(self, file_path: Path) -> ParseResult:
        from rag_parser.parsers.dedoc_tree import (
            build_full_tree,
            full_tree_to_markdown,
            normalize_markup_levels,
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

        # --- Docling: PDF и изображения ---
        # PDF и изображения ВСЕГДА через Docling (export_to_markdown →
        # существующий .md-путь). Никакого альтернативного парсера нет.
        if extension == ".pdf" or extension in IMAGE_EXTENSIONS:
            return self._parse_via_docling(path, size_bytes)

        self._validate_container(path, extension)
        require_libreoffice(extension, path.name)

        # Подготовка пути: .xls -> .xlsx своей конвертацией, .md -> .txt,
        # HTML -> HTML с маркерами таблиц (dedoc не знает markdown и не
        # даёт позиций таблиц в HTML).
        converted: Optional[Path] = None
        html_tables: List[str] = []
        effective_path = path
        effective_extension = extension
        if extension == ".xls":
            converted = convert_xls_to_xlsx(path)
            effective_path = converted
            effective_extension = ".xlsx"
        else:
            prepared = prepare_for_parsing(path)
            effective_path = prepared.path
            effective_extension = prepared.extension
            html_tables = prepared.html_tables
            converted = prepared.temp_path

        try:
            manager = prepare_manager_for_format(effective_extension)
        except ImportError as exc:
            if converted is not None:
                converted.unlink(missing_ok=True)
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
        # PDF и изображения больше не доходят сюда (обрабатываются
        # Docling-пайплайном _parse_via_docling).
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
        # У форматов с авторской разметкой тело документа начинается на уровень
        # глубже титульного блока (``<h2>`` вместо ``<h1>``). До промоции
        # приводим уровни к эталонной шкале, иначе пересборка вложенности
        # разложит все разделы на уровень ниже нужного.
        if extension in MARKUP_AUTHORITATIVE_EXTENSIONS:
            normalize_markup_levels(full_root)
        # Для форматов с авторской разметкой (h1-h6, <section>) типы и
        # уровни от dedoc авторитетны, и жёсткие правила только мешают:
        # они подняли бы нумерованный пункт «1 ПОДГОТОВЛЕН ...» (это
        # <item>, то есть абзац) до заголовка. У .txt разметки нет —
        # там заголовки распознаёт header_promoter.
        # Важно: смотрим на ИСХОДНОЕ расширение. Маршрут .md/.xml/.txt
        # отдаёт в dedoc временный .html, но авторитетность разметки
        # определяется форматом исходника.
        full_root = promote_headers(
            full_root,
            promote_non_headers=extension not in MARKUP_AUTHORITATIVE_EXTENSIONS,
        )
        # Привязка таблиц СТРОГО после промоции: якорь с таблицей-ребёнком
        # шёл бы composite-путём без резки строк (см. reattach_tables).
        # Таблицы без аннотаций (xlsx/csv) остаются orphan; для HTML-маршрута
        # позиция восстанавливается по маркерам.
        reattach_tables(full_root, tables)
        if html_tables:
            from rag_parser.parsers.dedoc_tree import attach_tables_by_position

            attach_tables_by_position(full_root, html_tables)
        markdown = full_tree_to_markdown(full_root)

        if not markdown.strip():
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

    def _parse_via_docling(self, path: Path, size_bytes: int) -> ParseResult:
        """PDF/изображение → Docling → Markdown → существующий ``.md``-путь.

        Извлечение содержимого выполняет только Docling. Полученный Markdown
        проходит прежний маршрут проекта: mistune → HTML → ``HtmlReader`` →
        дерево → заголовки/таблицы → чанкинг. Это структурирование, а не
        альтернативный парсинг.
        """
        from rag_parser.parsers.pdf_pipeline import run_pipeline

        extension = path.suffix.lower()
        self._validate_container(path, extension)

        try:
            pipeline_result = run_pipeline(path)
        except Exception as exc:  # noqa: BLE001 - упаковка ошибки Docling
            raise DocumentParseError(
                f"Docling processing failed for {path}: {exc}",
                filename=path.name,
            ) from exc

        markdown = pipeline_result.markdown
        if not markdown.strip():
            raise DocumentParseError(
                f"Docling produced no text for {path}", filename=path.name
            )

        # Уникальное имя: документы с одинаковым stem из разных каталогов
        # не должны перетирать друг друга.
        temp_dir = settings.ensure_temp_dir()
        md_path = temp_dir / f"docling_{uuid.uuid4().hex}_{path.stem}.md"
        try:
            md_path.write_text(markdown, encoding="utf-8")
        except OSError as exc:
            md_path.unlink(missing_ok=True)
            raise DocumentParseError(
                f"Cannot write temporary Markdown for {path}: {exc}",
                filename=path.name,
            ) from exc

        converted: Optional[Path] = None
        try:
            prepared = prepare_for_parsing(md_path)
            effective_path = prepared.path
            effective_extension = prepared.extension
            html_tables = prepared.html_tables
            converted = prepared.temp_path

            manager = prepare_manager_for_format(effective_extension)
            parsed = manager.parse(
                str(effective_path),
                parameters=parameters_for_format(effective_extension, effective_path),
            )
            document = parsed.to_api_schema()
            tree = document.model_dump()
        except (DocumentParseError, UnsupportedFormatError):
            raise
        except Exception as exc:  # noqa: BLE001 - ошибки dedoc
            raise DocumentParseError(
                f"dedoc failed to structure Docling Markdown for {path}: {exc}",
                filename=path.name,
            ) from exc
        finally:
            md_path.unlink(missing_ok=True)
            if converted is not None:
                converted.unlink(missing_ok=True)

        content = tree.get("content", {}) or {}
        tables = content.get("tables") or []
        structure = content.get("structure") or {}

        from rag_parser.parsers.dedoc_tree import (
            build_full_tree,
            full_tree_to_markdown,
            normalize_markup_levels,
            reattach_tables,
        )
        from rag_parser.parsers.header_promoter import promote_headers

        full_root = build_full_tree(structure, None)
        # Docling сам расставляет заголовки/списки/таблицы в Markdown,
        # поэтому его разметка авторитетна: не поднимаем абзацы до
        # заголовков жёсткими эвристиками. normalize_markup_levels лишь
        # выравнивает шкалу по нумерованным разделам.
        normalize_markup_levels(full_root)
        full_root = promote_headers(full_root, promote_non_headers=False)
        reattach_tables(full_root, tables)
        if html_tables:
            from rag_parser.parsers.dedoc_tree import attach_tables_by_position

            attach_tables_by_position(full_root, html_tables)

        final_markdown = full_tree_to_markdown(full_root)
        if not final_markdown.strip():
            raise DocumentParseError(
                f"Structured document from {path} contains no text",
                filename=path.name,
            )
        final_markdown = self._clean(final_markdown)

        document_id = self._new_document_id()
        paths = self.save_results(document_id, tree, final_markdown)
        pipeline_metadata = dict(pipeline_result.metadata)
        pipeline_metadata["format"] = self.format
        pipeline_metadata["extension"] = extension
        pipeline_metadata["size_bytes"] = size_bytes
        pipeline_metadata["parser"] = "docling"
        pipeline_metadata["dedoc_version"] = dedoc_version()
        pipeline_metadata["raw_tree_path"] = str(paths["raw_tree"])
        pipeline_metadata["markdown_path"] = str(paths["markdown"])
        pipeline_metadata["output_dir"] = str(paths["output_dir"])

        return ParseResult(
            document_id=document_id,
            filename=path.name,
            format=self.format,
            markdown=final_markdown,
            metadata=pipeline_metadata,
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

        output_dir = settings.ensure_dedoc_output_dir()
        tree_path = output_dir / f"{TREE_FILE_PREFIX}_{document_id}.json"
        markdown_path = output_dir / f"{MARKDOWN_FILE_PREFIX}_{document_id}.md"
        tree_path.write_bytes(orjson.dumps(tree, option=orjson.OPT_INDENT_2) + b"\n")
        markdown_path.write_bytes(markdown.encode("utf-8"))
        return {
            "output_dir": output_dir,
            "raw_tree": tree_path,
            "markdown": markdown_path,
        }


__all__ = [
    "DedocParser",
    "IMAGE_EXTENSIONS",
    "SUPPORTED_EXTENSIONS",
    "MARKDOWN_FILE_PREFIX",
    "TREE_FILE_PREFIX",
    "convert_xls_to_xlsx",
    "dedoc_version",
    "detect_text_encoding",
    "get_manager",
    "libreoffice_available",
    "parameters_for_format",
    "prepare_for_parsing",
    "prepare_manager_for_format",
    "require_libreoffice",
]
