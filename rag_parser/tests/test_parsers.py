"""Тесты единственного фасада — DedocParser (Docling + dedoc)."""

import json
import re
from pathlib import Path

import pytest

from rag_parser.core.config import settings
from rag_parser.core.interfaces import (
    DocumentParseError,
    UnsupportedFormatError,
)
from rag_parser.parsers.dedoc_parser import (
    MARKDOWN_FILE_PREFIX,
    SUPPORTED_EXTENSIONS,
    TREE_FILE_PREFIX,
    DedocParser,
    parameters_for_format,
)
from rag_parser.tests.conftest import _build_pdf, _text_ops

MARKDOWN_TABLE_RE = re.compile(r"^\|.*\|$", re.MULTILINE)
HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)


def assert_valid_markdown(markdown: str) -> None:
    assert isinstance(markdown, str) and markdown.strip()
    assert "\x00" not in markdown
    assert "\f" not in markdown
    assert "\r" not in markdown
    assert not re.search(r"\n{3,}", markdown)
    for line in markdown.splitlines():
        if line.startswith("|"):
            assert line.count("|") >= 2, f"malformed table row: {line!r}"


@pytest.fixture
def parser(tmp_path, monkeypatch):
    output_dir = tmp_path / "output" / "dedoc_results"
    monkeypatch.setattr(settings, "DEDOC_OUTPUT_DIR", str(output_dir))
    return DedocParser()


class TestSupported:
    def test_supported_formats(self):
        assert DedocParser().supported_extensions() == SUPPORTED_EXTENSIONS
        assert sorted(SUPPORTED_EXTENSIONS) == sorted(
            [".docx", ".doc", ".odt", ".rtf",
             ".xlsx", ".xls", ".csv", ".tsv",
             ".pdf", ".png", ".jpg", ".jpeg",
             ".html", ".htm", ".txt", ".xml", ".md"]
        )
        assert DedocParser.format == "dedoc"

    def test_format_count_matches_tz(self):
        # 13 расширений из ТЗ + .tsv (решение владельца) + 3 формата
        # изображений (png/jpg/jpeg) через Docling.
        assert len(SUPPORTED_EXTENSIONS) == 17

    def test_libreoffice_formats_declared(self):
        from rag_parser.parsers.dedoc_parser import LIBREOFFICE_EXTENSIONS

        assert set(LIBREOFFICE_EXTENSIONS) == {".doc", ".odt", ".rtf"}

    # test_parameters_pdf удалён: PDF и изображения не идут через dedoc
    # напрямую, их извлекает Docling (``_parse_via_docling``).

    def test_parameters_tsv(self, tmp_path):
        assert parameters_for_format(".tsv", tmp_path / "a.tsv")["delimiter"] == "\t"

    def test_parameters_csv_semicolon(self, tmp_path):
        sample = tmp_path / "a.csv"
        sample.write_text("a;b\n1;2\n", encoding="utf-8")
        assert parameters_for_format(".csv", sample)["delimiter"] == ";"

    def test_unsupported_raises_422_error(self, parser, tmp_path):
        path = tmp_path / "doc.pptx"
        path.write_bytes(b"fake")
        with pytest.raises(UnsupportedFormatError):
            parser.parse(path)


class TestParseFormats:
    def test_docx(self, parser, docx_files):
        result = parser.parse(docx_files["docx_headings"])
        assert result.format == "dedoc"
        assert "Заголовок документа" in result.markdown
        assert HEADING_RE.search(result.markdown)
        assert result.tree is not None

    def test_txt(self, parser, text_files):
        result = parser.parse(text_files["txt_utf8"])
        assert "Первая строка документа." in result.markdown
        assert_valid_markdown(result.markdown)

    def test_html(self, parser, html_files):
        result = parser.parse(html_files["html_full"])
        assert "Заголовок" in result.markdown
        assert_valid_markdown(result.markdown)

    def test_htm(self, parser, html_files):
        result = parser.parse(html_files["htm"])
        assert "Заголовок" in result.markdown

    def test_csv(self, parser, csv_files):
        result = parser.parse(csv_files["csv_semicolon"])
        assert "Значение1" in result.markdown
        assert MARKDOWN_TABLE_RE.search(result.markdown)

    def test_tsv(self, parser, csv_files):
        result = parser.parse(csv_files["tsv"])
        assert "banana" in result.markdown

    def test_xlsx(self, parser, xlsx_files):
        result = parser.parse(xlsx_files["xlsx_multi"])
        assert "Колонка1" in result.markdown
        assert MARKDOWN_TABLE_RE.search(result.markdown)

    def test_xls(self, parser, xls_files):
        result = parser.parse(xls_files["xls"])
        assert "row-0" in result.markdown

    def test_xlsx_goes_through_dedoc_without_conversion(self, parser, xlsx_files,
                                                        tmp_path, monkeypatch):
        """Фасад: .xlsx — напрямую в DedocManager.parse, без xlrd-конвертации."""
        import rag_parser.parsers.dedoc_parser as dedoc_module
        from rag_parser.core.config import settings

        monkeypatch.setattr(
            settings, "DEDOC_OUTPUT_DIR", str(tmp_path / "output" / "dedoc_results")
        )

        def _forbid_conversion(source):
            raise AssertionError("convert_xls_to_xlsx must not run for .xlsx")

        monkeypatch.setattr(
            dedoc_module, "convert_xls_to_xlsx", _forbid_conversion
        )
        calls = []
        real_parse = dedoc_module.get_manager().parse

        def _spy_parse(file_path, parameters=None):
            calls.append((file_path, parameters))
            return real_parse(file_path, parameters=parameters)

        monkeypatch.setattr(dedoc_module.get_manager(), "parse", _spy_parse)
        result = DedocParser().parse(xlsx_files["xlsx_multi"])
        assert result.markdown.strip()
        assert len(calls) == 1  # DedocManager.parse вызван ровно один раз
        assert str(calls[0][0]).endswith(".xlsx")

    def test_xls_uses_conversion_inside_facade(self, parser, xls_files, tmp_path,
                                               monkeypatch):
        """Фасад: .xls — конвертация внутри DedocParser (soffice запрещён)."""
        import rag_parser.parsers.dedoc_parser as dedoc_module
        from rag_parser.core.config import settings

        monkeypatch.setattr(
            settings, "DEDOC_OUTPUT_DIR", str(tmp_path / "output" / "dedoc_results")
        )
        calls = []
        real_convert = dedoc_module.convert_xls_to_xlsx

        def _spy_convert(source):
            calls.append(source)
            return real_convert(source)

        monkeypatch.setattr(dedoc_module, "convert_xls_to_xlsx", _spy_convert)
        result = DedocParser().parse(xls_files["xls"])
        assert len(calls) == 1
        assert "row-0" in result.markdown

    def test_pdf_text(self, parser, pdf_files, monkeypatch):
        # PDF идёт через Docling. Модели в юнит-тестах не загружаются —
        # мокаем run_pipeline. Реальная конвертация проверяется отдельно.
        from rag_parser.parsers import pdf_pipeline
        from rag_parser.parsers.pdf_pipeline import PdfPipelineResult

        def _mock(path, **kwargs):
            return PdfPipelineResult(
                markdown="# PDF document\n\nDocling mocked body.",
                metadata={"source_pipeline": "docling", "input_format": "pdf",
                          "pages": 1, "docling_status": "success",
                          "extension": ".pdf", "size_bytes": 100,
                          "parser": "docling"},
            )

        monkeypatch.setattr(pdf_pipeline, "run_pipeline", _mock)
        result = parser.parse(pdf_files["pdf_text"])
        assert result.markdown.strip()
        assert result.metadata["parser"] == "docling"

    def test_image_via_docling(self, parser, tmp_path, monkeypatch):
        from PIL import Image

        from rag_parser.parsers import pdf_pipeline
        from rag_parser.parsers.pdf_pipeline import PdfPipelineResult

        source = tmp_path / "scan.png"
        Image.new("RGB", (40, 20), "white").save(source)
        calls = []

        def _mock(path, **kwargs):
            calls.append(path)
            return PdfPipelineResult(
                markdown="# Скан\n\nТекст изображения.",
                metadata={"source_pipeline": "docling", "input_format": "image",
                          "pages": 1, "parser": "docling"},
            )

        monkeypatch.setattr(pdf_pipeline, "run_pipeline", _mock)
        result = parser.parse(source)

        assert calls == [source]
        assert result.metadata["parser"] == "docling"
        assert result.metadata["extension"] == ".png"
        assert "Текст изображения" in result.markdown

    def test_markdown_write_failure_is_reported(
        self, parser, tmp_path, monkeypatch
    ):
        from rag_parser.parsers import pdf_pipeline
        from rag_parser.parsers.pdf_pipeline import PdfPipelineResult

        source = tmp_path / "doc.pdf"
        source.write_bytes(b"%PDF-1.4\n%%EOF\n")
        monkeypatch.setattr(
            pdf_pipeline,
            "run_pipeline",
            lambda path: PdfPipelineResult(
                markdown="# Header\n\nBody", metadata={"parser": "docling"}
            ),
        )

        def _boom(self, *args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", _boom)
        with pytest.raises(DocumentParseError, match="Cannot write temporary Markdown"):
            parser.parse(source)

    def test_header_recognition_is_deterministic(
        self, header_docs, tmp_path, monkeypatch
    ):
        """Нет порога: результат распознавания заголовков стабилен."""
        monkeypatch.setattr(
            settings, "DEDOC_OUTPUT_DIR", str(tmp_path / "output" / "dedoc_results")
        )
        from rag_parser.parsers.dedoc_tree import collect_headers

        first = DedocParser().parse(header_docs["unstyled"])
        second = DedocParser().parse(header_docs["unstyled"])
        assert [h.text for h in collect_headers(first.tree)] == [
            h.text for h in collect_headers(second.tree)
        ]
        assert collect_headers(first.tree), "хотя бы один заголовок должен найден"


class TestTree:
    def test_tree_shape(self, parser, docx_files):
        result = parser.parse(docx_files["docx_headings"])
        assert result.tree is not None
        assert result.tree.type == "document"
        headers = [
            node for node in _walk(result.tree) if node.type == "header"
        ]
        assert headers
        for header in headers:
            assert header.level >= 1
            assert header.text.strip()
        # урезание до 30 символов — последний шаг в API/UI, не в парсере
        from rag_parser.parsers.dedoc_tree import truncate

        preview = truncate(result.tree)
        for node in _walk(preview):
            assert len(node.text) <= 31
            assert "annotations" not in node.metadata
            assert "annotations_count" in node.metadata

    def test_full_text_in_markdown_not_truncated(self, parser, docx_files):
        result = parser.parse(docx_files["docx_headings"])
        long_line = "Первый абзац текста."
        assert long_line in result.markdown

    def test_results_saved(self, parser, complex_docx, tmp_path):
        result = parser.parse(complex_docx)
        tree_path = Path(result.metadata["raw_tree_path"])
        markdown_path = Path(result.metadata["markdown_path"])
        output_dir = tmp_path / "output" / "dedoc_results"
        assert tree_path == output_dir / f"{TREE_FILE_PREFIX}_{result.document_id}.json"
        assert markdown_path == output_dir / f"{MARKDOWN_FILE_PREFIX}_{result.document_id}.md"
        assert tree_path.exists() and markdown_path.exists()
        tree = json.loads(tree_path.read_text(encoding="utf-8"))
        assert tree["content"]["structure"]
        assert markdown_path.read_text(encoding="utf-8") == result.markdown

    def test_unique_ids(self, parser, complex_docx):
        assert parser.parse(complex_docx).document_id != parser.parse(complex_docx).document_id

    def test_table_attached_to_anchor(self, parser, tmp_path):
        """Раздел 1 → абзац → таблица → Раздел 2: таблица под Разделом 1."""
        from docx import Document

        doc = Document()
        doc.add_paragraph("Раздел 1. Первый раздел документа")
        doc.add_paragraph("Абзац с данными в таблице ниже.")
        table = doc.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Метрика"
        table.cell(0, 1).text = "Значение"
        table.cell(1, 0).text = "X"
        table.cell(1, 1).text = "10"
        doc.add_paragraph("Раздел 2. Второй раздел документа")
        path = tmp_path / "anchored.docx"
        doc.save(path)

        result = parser.parse(path)
        parents = {}
        ordered = []
        stack = [(result.tree, None)]
        while stack:
            node, parent = stack.pop(0)
            parents[id(node)] = parent
            ordered.append(node)
            stack.extend((child, node) for child in node.children)
        table_node = next(
            n for n in ordered
            if n.type == "table" and "Метрика" in (n.text or "")
        )
        chain = []
        parent = parents[id(table_node)]
        while parent is not None and parent.type != "document":
            chain.append(parent)
            parent = parents[id(parent)]
        # якорь-абзац — прямой родитель, выше — заголовок Раздела 1
        assert chain[0].type in ("paragraph", "raw_text")
        assert any(n.type == "header" and n.text.startswith("Раздел 1.")
                   for n in chain)
        assert not any(n.type == "header" and n.text.startswith("Раздел 2.")
                        for n in chain)
        pos_table = result.markdown.index("| Метрика | Значение |")
        pos_b = result.markdown.index("Раздел 2.")
        assert pos_table < pos_b  # не под Разделом 2 и не в конце


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


class TestErrors:
    def test_missing(self, parser, tmp_path):
        with pytest.raises(DocumentParseError, match="not found"):
            parser.parse(tmp_path / "missing.docx")

    def test_empty(self, parser, tmp_path):
        path = tmp_path / "empty.docx"
        path.write_bytes(b"")
        with pytest.raises(DocumentParseError, match="empty"):
            parser.parse(path)

    def test_broken_docx(self, parser, invalid_files):
        with pytest.raises(DocumentParseError):
            parser.parse(invalid_files["broken_docx"])

    def test_doc_rejected(self, parser, tmp_path):
        path = tmp_path / "old.pptx"
        path.write_bytes(b"fake")
        with pytest.raises(UnsupportedFormatError):
            parser.parse(path)


class TestNewFormats:
    """Форматы, добавленные в Этапе 1: .doc/.odt/.rtf/.xml/.md."""

    INPUT_DIR = Path(__file__).resolve().parent.parent.parent / "input"

    def _require(self, name: str) -> Path:
        path = self.INPUT_DIR / name
        if not path.exists():
            pytest.skip(f"{name} отсутствует в input/")
        return path

    @pytest.mark.parametrize(
        "name", ["gost_report.doc", "gost_report.odt", "gost_report.rtf"]
    )
    def test_libreoffice_formats(self, parser, name):
        result = parser.parse(self._require(name))
        assert result.markdown.strip()
        assert_valid_markdown(result.markdown)

    @pytest.mark.parametrize(
        "name", ["gost_report.docx", "gost_report.odt", "gost_report.rtf"]
    )
    def test_word_family_identical_output(self, parser, name):
        """DOCX/ODT/RTF — один документ в трёх обёртках: результат совпадает."""
        reference = parser.parse(self._require("gost_report.docx")).markdown
        result = parser.parse(self._require(name)).markdown
        assert result == reference

    def test_markdown_alias_copied_to_txt(self, parser, monkeypatch, tmp_path):
        """.md больше не идёт в .txt: конвертируется в HTML (mistune).

        Исторически .md копировался в .txt и уходил в RawTextReader, откуда
        # и «# Заголовок», и pipe-таблица приходили обычным текстом (все
        чанки с одним parent_id). Теперь markdown -> HTML -> HtmlReader.
        """
        from rag_parser.parsers.dedoc_parser import prepare_for_parsing

        source = self._require("gost_report.md")
        prepared = prepare_for_parsing(source)
        try:
            assert prepared.extension == ".html"
            assert prepared.temp_path is not None and prepared.temp_path.exists()
            assert prepared.temp_path.suffix == ".html"
            # таблицы извлечены нами и пойдут по маркерному маршруту
            assert len(prepared.html_tables) == 9

            html = prepared.temp_path.read_text(encoding="utf-8")
            assert "<h1>" in html and "<table>" not in html
            assert "__RAGTABLE_0__" in html
        finally:
            if prepared.temp_path is not None:
                prepared.temp_path.unlink(missing_ok=True)

    def test_markdown_parses(self, parser):
        result = parser.parse(self._require("gost_report.md"))
        assert result.markdown.strip()
        assert_valid_markdown(result.markdown)

    def test_xml_parses(self, parser):
        result = parser.parse(self._require("gost_report.xml"))
        assert result.markdown.strip()
        assert_valid_markdown(result.markdown)

    def test_odt_returns_tables(self, parser):
        result = parser.parse(self._require("gost_report.odt"))
        assert result.metadata["tables"] >= 1

    def test_temp_alias_file_cleaned_up(self, parser):
        """Временная .txt-копия не должна оставаться в temp."""
        from rag_parser.core.config import settings

        temp_dir = Path(settings.TEMP_DIR)
        stem = "gost_report"
        leftovers = (
            list(temp_dir.glob(f"{stem}.txt")) if temp_dir.exists() else []
        )
        parser.parse(self._require("gost_report.md"))
        remaining = (
            [p for p in temp_dir.glob(f"{stem}.txt") if p not in leftovers]
            if temp_dir.exists()
            else []
        )
        assert not remaining, f"временные файлы не удалены: {remaining}"


class TestLibreOfficeGuard:
    def test_available_flag_matches_path(self):
        import shutil

        from rag_parser.parsers.dedoc_parser import libreoffice_available

        expected = (
            shutil.which("soffice") is not None
            or shutil.which("libreoffice") is not None
        )
        assert libreoffice_available() is expected

    def test_missing_libreoffice_gives_clear_error(self, monkeypatch, tmp_path):
        from rag_parser.parsers import dedoc_parser as module

        monkeypatch.setattr(module.shutil, "which", lambda name: None)
        path = tmp_path / "doc.odt"
        path.write_bytes(b"fake")
        with pytest.raises(DocumentParseError, match="LibreOffice"):
            module.require_libreoffice(".odt", path.name)

    def test_guard_silent_for_non_libreoffice_formats(self, monkeypatch):
        from rag_parser.parsers import dedoc_parser as module

        monkeypatch.setattr(module.shutil, "which", lambda name: None)
        for extension in (".docx", ".pdf", ".txt", ".md", ".xlsx"):
            module.require_libreoffice(extension, "any.bin")

    def test_container_signature_rejects_fake_odt(self, parser, tmp_path):
        path = tmp_path / "fake.odt"
        path.write_bytes(b"not a zip container at all")
        with pytest.raises(DocumentParseError, match="does not match"):
            parser.parse(path)

    def test_container_signature_rejects_fake_doc(self, parser, tmp_path):
        path = tmp_path / "fake.doc"
        path.write_bytes(b"not an OLE2 container")
        with pytest.raises(DocumentParseError, match="does not match"):
            parser.parse(path)
