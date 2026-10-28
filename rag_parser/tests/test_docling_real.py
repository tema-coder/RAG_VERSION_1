"""Реальная конвертация Docling (не входит в обычный прогон).

Запуск (скачает модели Docling при первом вызове):

    RAG_REAL_DOCLING=1 python -m pytest rag_parser/tests/test_docling_real.py -q

В обычном ``pytest`` модуль пропускается: загрузка layout/table/OCR-моделей
тяжёлая и требует сети.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from rag_parser.parsers.docling_converter import convert_to_markdown
from rag_parser.tests.conftest import _build_pdf, _table_ops, pdf_page

pytestmark = pytest.mark.skipif(
    os.getenv("RAG_REAL_DOCLING") != "1",
    reason="set RAG_REAL_DOCLING=1 to run real Docling conversion",
)


def test_real_text_pdf(tmp_path: Path):
    path = _build_pdf(
        [
            pdf_page("First page of the docling test document."),
            pdf_page("Second page with more text."),
        ],
        None,
        tmp_path / "text.pdf",
    )
    result = convert_to_markdown(path)
    markdown = result.markdown.lower()
    assert "first page" in markdown
    assert "second page" in markdown
    assert result.page_count == 2


def test_real_pdf_with_table(tmp_path: Path):
    path = _build_pdf(
        [pdf_page("Quarterly report")],
        [[["Region", "Q1"], ["North", "100"], ["South", "90"]]],
        tmp_path / "table.pdf",
    )
    result = convert_to_markdown(path)
    markdown = result.markdown
    assert "North" in markdown
    assert "100" in markdown
    # Docling экспортирует таблицу в pipe-Markdown.
    assert "|" in markdown


def test_real_png_image(tmp_path: Path):
    path = tmp_path / "scan.png"
    image = Image.new("RGB", (900, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 100), "Docling OCR test 12345", fill="black")
    image.save(path)

    result = convert_to_markdown(path)
    assert result.source == "image"
    assert result.markdown.strip()
    assert "Docling" in result.markdown or "OCR" in result.markdown


def test_real_jpg_image(tmp_path: Path):
    path = tmp_path / "scan.jpg"
    image = Image.new("RGB", (900, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 100), "JPEG OCR test 67890", fill="black")
    image.save(path, format="JPEG")

    result = convert_to_markdown(path)
    assert result.source == "image"
    assert result.markdown.strip()
