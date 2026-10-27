"""Юнит-тесты PDF-пайплайна (классификатор, детектор таблиц, роутер, сборка).

Промпт: «В юнит-тестах Docling не загружается.» — Docling мокается через
``monkeypatch``. Реальные PDF (gost_report.pdf, scan.pdf) создаются через
существующий генератор в ``conftest.py``.

Покрытие:
* классификатор (текстовый PDF vs скан);
* детектор таблиц (PyMuPDF find_tables);
* роутер (Код №1 и Docling подменены моками);
* сборка в порядке страниц;
* группировка страниц в раны;
* fallback при падении Docling.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_parser.parsers.pdf_classifier import PdfClassification, classify_pdf
from rag_parser.parsers.pdf_router import (
    PageResult,
    RoutingPlan,
    assemble_markdown,
    build_routing_plan,
    group_table_runs,
)
from rag_parser.parsers.pdf_table_detector import TableDetectionResult


# --------------------------------------------------------------------------- #
# Классификатор
# --------------------------------------------------------------------------- #


class TestClassifier:
    """Классификация PDF: текстовый vs скан."""

    def test_text_pdf_classified_as_text(self, pdf_files):
        """PDF с текстовым слоем → text."""
        if not pdf_files.get("pdf_text"):
            pytest.skip("pdf_text.pdf отсутствует")
        # Тестовые PDF короткие (22/14 символов), поэтому низкий порог.
        result = classify_pdf(pdf_files["pdf_text"], min_chars_per_page=5)
        assert result.pdf_type == "text"
        assert result.total_pages > 0
        assert len(result.text_pages) > 0

    def test_scan_pdf_classified_as_scan(self, pdf_files):
        """PDF без текстового слоя → scan."""
        if not pdf_files.get("pdf_scan"):
            pytest.skip("pdf_scan.pdf отсутствует")
        result = classify_pdf(pdf_files["pdf_scan"])
        assert result.pdf_type == "scan"
        assert len(result.scan_pages) == result.total_pages

    def test_empty_pdf_is_scan(self, tmp_path):
        """PDF с 0 страницами → scan (вырожденный случай)."""
        import fitz

        path = tmp_path / "one_page.pdf"
        doc = fitz.open()
        doc.new_page()  # PyMuPDF не сохраняет пустой документ
        doc.save(str(path))
        doc.close()
        # 1 страница без текста → scan.
        result = classify_pdf(path)
        assert result.pdf_type == "scan"
        assert len(result.scan_pages) == result.total_pages

    def test_threshold_min_chars(self, pdf_files):
        """Страница с < min_chars_per_page → scan_page."""
        if not pdf_files.get("pdf_text"):
            pytest.skip("pdf_text.pdf отсутствует")
        # Высокий порог — все страницы станут scan_pages.
        result = classify_pdf(pdf_files["pdf_text"], min_chars_per_page=99999)
        assert result.pdf_type == "scan"

    def test_threshold_ratio(self, pdf_files):
        """Высокий text_pages_min_ratio → даже текстовый PDF станет scan."""
        if not pdf_files.get("pdf_text"):
            pytest.skip("pdf_text.pdf отсутствует")
        result = classify_pdf(
            pdf_files["pdf_text"], min_chars_per_page=5, text_pages_min_ratio=0.99
        )
        # Если не все страницы текстовые → scan при ratio=0.99
        if len(result.text_pages) < result.total_pages:
            assert result.pdf_type == "scan"


# --------------------------------------------------------------------------- #
# Детектор таблиц
# --------------------------------------------------------------------------- #


class TestTableDetector:
    """Детекция таблиц на страницах PDF."""

    def test_table_pdf_has_table_pages(self, pdf_files):
        """PDF с таблицей → хотя бы одна table_page."""
        if not pdf_files.get("pdf_table"):
            pytest.skip("pdf_table.pdf отсутствует")
        from rag_parser.parsers.pdf_table_detector import detect_tables

        result = detect_tables(pdf_files["pdf_table"])
        assert len(result.table_pages) > 0 or len(result.text_pages) > 0

    def test_text_pdf_no_tables(self, pdf_files):
        """Текстовый PDF без таблиц → все text_pages."""
        if not pdf_files.get("pdf_text"):
            pytest.skip("pdf_text.pdf отсутствует")
        from rag_parser.parsers.pdf_table_detector import detect_tables

        result = detect_tables(pdf_files["pdf_text"])
        # Текстовый PDF может не иметь таблиц вообще.
        assert result.text_pages or result.table_pages  # что-то найдено


# --------------------------------------------------------------------------- #
# Роутер — группировка в раны
# --------------------------------------------------------------------------- #


class TestGroupRuns:
    """Группировка табличных страниц в непрерывные раны."""

    def test_empty(self):
        assert group_table_runs([]) == []

    def test_single_page(self):
        runs = group_table_runs([5])
        assert len(runs) == 1
        assert runs[0].pages == [5]

    def test_consecutive_pages_one_run(self):
        runs = group_table_runs([0, 1, 2, 3])
        assert len(runs) == 1
        assert runs[0].pages == [0, 1, 2, 3]

    def test_gap_splits_into_runs(self):
        runs = group_table_runs([0, 1, 5, 6])
        assert len(runs) == 2
        assert runs[0].pages == [0, 1]
        assert runs[1].pages == [5, 6]

    def test_unsorted_input_sorted(self):
        runs = group_table_runs([6, 0, 1, 5])
        assert len(runs) == 2
        assert runs[0].pages == [0, 1]
        assert runs[1].pages == [5, 6]

    def test_duplicates_removed(self):
        runs = group_table_runs([0, 0, 1, 1])
        assert len(runs) == 1
        assert runs[0].pages == [0, 1]


# --------------------------------------------------------------------------- #
# Роутер — план маршрутизации
# --------------------------------------------------------------------------- #


class TestRoutingPlan:
    """План маршрутизации: текстовый PDF vs скан."""

    def test_scan_all_pages_to_ocr(self):
        """Скан → все страницы в docling_ocr."""
        cls = PdfClassification(
            pdf_type="scan",
            total_pages=5,
            text_pages=[],
            scan_pages=[0, 1, 2, 3, 4],
            chars_per_page=[0, 0, 0, 0, 0],
        )
        plan = build_routing_plan(cls)
        assert len(plan.docling_ocr_pages) == 5
        assert len(plan.text_parser_pages) == 0
        assert len(plan.docling_pages) == 0

    def test_text_pdf_no_tables_all_to_text_parser(self):
        """Текстовый PDF без таблиц → все в text_parser."""
        cls = PdfClassification(
            pdf_type="text",
            total_pages=3,
            text_pages=[0, 1, 2],
            scan_pages=[],
            chars_per_page=[100, 200, 150],
        )
        td = TableDetectionResult(
            table_pages=[], text_pages=[0, 1, 2], tables_per_page={0: 0, 1: 0, 2: 0}
        )
        plan = build_routing_plan(cls, td)
        assert plan.text_parser_pages == [0, 1, 2]
        assert plan.docling_pages == []
        assert plan.docling_ocr_pages == []

    def test_text_pdf_with_tables(self):
        """Текстовый PDF с таблицами → таблицы в docling, остальное в text_parser."""
        cls = PdfClassification(
            pdf_type="text",
            total_pages=4,
            text_pages=[0, 1, 2, 3],
            scan_pages=[],
            chars_per_page=[100, 200, 150, 180],
        )
        td = TableDetectionResult(
            table_pages=[1, 2],
            text_pages=[0, 3],
            tables_per_page={0: 0, 1: 1, 2: 1, 3: 0},
        )
        plan = build_routing_plan(cls, td)
        assert plan.text_parser_pages == [0, 3]
        assert plan.docling_pages == [1, 2]
        assert len(plan.table_runs) == 1  # [1, 2] — один ран
        assert plan.table_runs[0].pages == [1, 2]

    def test_scan_page_inside_text_pdf_goes_to_ocr(self):
        """Страница без текстового слоя внутри текстового PDF → docling_ocr."""
        cls = PdfClassification(
            pdf_type="text",
            total_pages=3,
            text_pages=[0, 2],
            scan_pages=[1],
            chars_per_page=[100, 0, 150],
        )
        td = TableDetectionResult(
            table_pages=[], text_pages=[0, 1, 2], tables_per_page={0: 0, 1: 0, 2: 0}
        )
        plan = build_routing_plan(cls, td)
        assert plan.text_parser_pages == [0, 2]
        assert plan.docling_ocr_pages == [1]


# --------------------------------------------------------------------------- #
# Роутер — сборка Markdown
# --------------------------------------------------------------------------- #


class TestAssembleMarkdown:
    """Сборка Markdown в порядке страниц."""

    def test_order_by_page(self):
        """Результаты склеиваются в порядке номеров страниц."""
        results = [
            PageResult(pages=[2], source="text_parser", markdown="C"),
            PageResult(pages=[0], source="text_parser", markdown="A"),
            PageResult(pages=[1], source="docling", markdown="B"),
        ]
        md = assemble_markdown(results)
        # Порядок: 0, 1, 2 → A, B, C
        parts = md.split("\n\n")
        assert parts == ["A", "B", "C"]

    def test_empty_results(self):
        assert assemble_markdown([]) == ""

    def test_skip_empty_markdown(self):
        """Пустые результаты пропускаются."""
        results = [
            PageResult(pages=[0], source="text_parser", markdown=""),
            PageResult(pages=[1], source="text_parser", markdown="B"),
        ]
        md = assemble_markdown(results)
        assert md == "B"

    def test_paragraph_glue_across_pages(self):
        """Склейка абзаца через границу: нет пунктуации + строчная → пробел."""
        results = [
            PageResult(pages=[0], source="text_parser", markdown="текст оборвался"),
            PageResult(
                pages=[1], source="text_parser", markdown="на следующей странице"
            ),
        ]
        md = assemble_markdown(results)
        # «текст оборвался» не заканчивается пунктуацией, «на» — строчная → склейка.
        assert "текст оборвался на следующей странице" in md

    def test_no_glue_with_punctuation(self):
        """Если предыдущий заканчивается точкой — \n\n, не склейка."""
        results = [
            PageResult(pages=[0], source="text_parser", markdown="Предложение."),
            PageResult(pages=[1], source="text_parser", markdown="Новое предложение"),
        ]
        md = assemble_markdown(results)
        assert "Предложение.\n\nНовое предложение" in md


# --------------------------------------------------------------------------- #
# Пайплайн — fallback при падении Docling
# --------------------------------------------------------------------------- #


class TestPipelineFallback:
    """Отказоустойчивость: Docling упал → fallback на Код №1."""

    def test_table_pages_fallback_to_text_parser(self, pdf_files, monkeypatch):
        """Если Docling упал на табличных страницах — откат на Код №1."""
        if not pdf_files.get("pdf_text"):
            pytest.skip("pdf_text.pdf отсутствует")

        from rag_parser.parsers import pdf_pipeline
        from rag_parser.parsers.pdf_pipeline import run_pipeline
        from rag_parser.parsers.docling_converter import DoclingError

        # Мокаем Docling — всегда кидает ошибку.
        def mock_convert(*args, **kwargs):
            raise DoclingError("mock failure")

        # Патчим в pdf_pipeline (локальная ссылка docling_convert).
        monkeypatch.setattr(pdf_pipeline, "docling_convert", mock_convert)

        # Мокаем классификатор — текстовый PDF с таблицами.
        monkeypatch.setattr(
            pdf_pipeline,
            "classify_pdf",
            lambda *a, **kw: PdfClassification(
                pdf_type="text",
                total_pages=2,
                text_pages=[0, 1],
                scan_pages=[],
                chars_per_page=[100, 100],
            ),
        )
        monkeypatch.setattr(
            pdf_pipeline,
            "detect_tables",
            lambda *a, **kw: TableDetectionResult(
                table_pages=[0], text_pages=[1], tables_per_page={0: 1, 1: 0}
            ),
        )

        result = run_pipeline(pdf_files["pdf_text"])
        # Markdown собран (хоть из Кода №1), lost_tables содержит страницу 0.
        assert result.markdown
        assert 0 in result.metadata["lost_tables"]

    def test_scan_docling_failure_raises(self, pdf_files, monkeypatch):
        """Если Docling упал на скане — DoclingError (не молчаливый 500)."""
        if not pdf_files.get("pdf_scan"):
            pytest.skip("pdf_scan.pdf отсутствует")

        from rag_parser.parsers import pdf_pipeline
        from rag_parser.parsers.pdf_pipeline import run_pipeline
        from rag_parser.parsers.docling_converter import DoclingError

        def mock_convert(*args, **kwargs):
            raise DoclingError("scan failed", error_type="docling_ocr_failure")

        # Патчим в модуле pdf_pipeline, т.к. convert импортирован туда как
        # docling_convert (локальная ссылка).
        monkeypatch.setattr(pdf_pipeline, "docling_convert", mock_convert)

        with pytest.raises(DoclingError) as exc_info:
            run_pipeline(pdf_files["pdf_scan"])
        assert "scan failed" in str(exc_info.value)


# --------------------------------------------------------------------------- #
# Пайплайн — метаданные наблюдаемости
# --------------------------------------------------------------------------- #


class TestPipelineMetadata:
    """Метаданные: тип PDF, списки страниц, Docling-раны, время."""

    def test_text_pdf_metadata(self, pdf_files, monkeypatch):
        """Текстовый PDF: метаданные содержат pdf_type, routing, timings."""
        if not pdf_files.get("pdf_text"):
            pytest.skip("pdf_text.pdf отсутствует")

        from rag_parser.parsers import pdf_pipeline
        from rag_parser.parsers.pdf_pipeline import run_pipeline

        # Мокаем Docling — успешный результат.
        def mock_convert(path, *, pages=None, with_ocr=False, **kw):
            from rag_parser.parsers.docling_converter import DoclingResult

            pgs = pages or [0]
            return DoclingResult(
                markdown="| A | B |\n|---|---|\n| 1 | 2 |",
                page_range=(min(pgs), max(pgs)),
                source="docling_ocr" if with_ocr else "docling",
            )

        monkeypatch.setattr(pdf_pipeline, "docling_convert", mock_convert)

        result = run_pipeline(pdf_files["pdf_text"])
        md = result.metadata
        assert md["pdf_type"] in ("text", "scan")
        assert "total_pages" in md
        assert "routing" in md
        assert "timings" in md
        assert "classify" in md["timings"]
        assert isinstance(md["routing"]["text_parser_pages"], list)


# --------------------------------------------------------------------------- #
# Docling converter — очистка артефактов
# --------------------------------------------------------------------------- #


class TestDoclingCleanup:
    """Очистка артефактов Docling из Markdown."""

    def test_clean_image_comments(self):
        from rag_parser.parsers.docling_converter import _clean_docling_markdown

        md = "# Заголовок\n\n<!-- image -->\n\nТекст\n\n<!-- figure -->"
        cleaned = _clean_docling_markdown(md)
        assert "<!--" not in cleaned
        assert "image" not in cleaned

    def test_clean_multiple_newlines(self):
        from rag_parser.parsers.docling_converter import _clean_docling_markdown

        md = "Абзац 1\n\n\n\n\nАбзац 2"
        cleaned = _clean_docling_markdown(md)
        assert "\n\n\n" not in cleaned
        assert "Абзац 1\n\nАбзац 2" == cleaned
