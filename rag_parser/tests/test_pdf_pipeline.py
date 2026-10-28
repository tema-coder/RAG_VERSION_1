"""Юнит-тесты Docling-пайплайна PDF/изображений.

Модели Docling в юнит-тестах не загружаются: подменяется
``docling_converter.convert_to_markdown``. Реальная конвертация проверяется
отдельно в ``test_docling_real.py`` (требует скачанных моделей).
"""

from __future__ import annotations

import pytest

from rag_parser.parsers import pdf_pipeline
from rag_parser.parsers.docling_converter import DoclingError, DoclingResult


class TestRunPipeline:
    def test_pdf_result_is_mapped(self, tmp_path, monkeypatch):
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"%PDF-1.4\n")
        calls = []

        def _fake(path):
            calls.append(path)
            return DoclingResult(
                markdown="# Заголовок\n\nТекст документа.",
                page_count=2,
                source="pdf",
                metadata={"docling_status": "success"},
            )

        monkeypatch.setattr(
            pdf_pipeline.docling_converter, "convert_to_markdown", _fake
        )
        result = pdf_pipeline.run_pipeline(source)

        assert calls == [source]
        assert result.markdown.startswith("# Заголовок")
        assert result.metadata["source_pipeline"] == "docling"
        assert result.metadata["input_format"] == "pdf"
        assert result.metadata["pages"] == 2
        assert result.metadata["docling_status"] == "success"

    def test_image_result_is_mapped(self, tmp_path, monkeypatch):
        source = tmp_path / "scan.png"
        source.write_bytes(b"\x89PNG\r\n\x1a\n")

        monkeypatch.setattr(
            pdf_pipeline.docling_converter,
            "convert_to_markdown",
            lambda path: DoclingResult(
                markdown="Текст со скана",
                page_count=1,
                source="image",
                metadata={"source_extension": ".png"},
            ),
        )
        result = pdf_pipeline.run_pipeline(source)

        assert result.metadata["input_format"] == "image"
        assert result.metadata["source_extension"] == ".png"

    def test_parse_pdf_returns_markdown(self, tmp_path, monkeypatch):
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"%PDF-1.4\n")
        monkeypatch.setattr(
            pdf_pipeline.docling_converter,
            "convert_to_markdown",
            lambda path: DoclingResult(markdown="Проверка", page_count=1),
        )
        assert pdf_pipeline.parse_pdf(source) == "Проверка"

    def test_docling_error_is_not_hidden(self, tmp_path, monkeypatch):
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"%PDF-1.4\n")

        def _boom(path):
            raise DoclingError("Docling failed", error_type="docling_failure")

        monkeypatch.setattr(
            pdf_pipeline.docling_converter, "convert_to_markdown", _boom
        )
        with pytest.raises(DoclingError):
            pdf_pipeline.run_pipeline(source)

    def test_no_alternative_parser_symbols(self):
        """В пайплайне не осталось символов старых парсеров."""
        source = __import__("inspect").getsource(pdf_pipeline)
        for forbidden in (
            "pdf_classifier",
            "pdf_router",
            "pdf_text_parser",
            "pdf_page_processor",
            "easyocr",
            "fitz",
            "process_complex_page",
        ):
            assert forbidden not in source
