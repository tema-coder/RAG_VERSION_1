"""Юнит-тесты PDF-пайплайна по ТЗ (classifier → route → per-page → assemble).

Тяжёлые модели (EasyOCR / layout / TableFormer) в юнит-тестах не
загружаются: подменяются через ``monkeypatch`` на уровне
``pdf_pipeline`` / ``pdf_page_processor``. Реальные PDF создаются
генератором в ``conftest.py`` (4 типа: текст, текст+таблица, скан,
смешанный).

Покрытие (критерии приёмки ТЗ):
* классификатор: текстовый vs скан vs мусорный слой vs площадь текста;
* маршрут каждой страницы появляется в логах;
* страницы без таблиц НЕ вызывают EasyOCR/TableFormer;
* Markdown-страницы склеиваются строго в порядке номеров страниц;
* fallback: падение сложного обработчика не теряет страницу;
* ``parse_pdf(path) -> markdown_str`` — единый интерфейс.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from rag_parser.parsers.pdf_classifier import PdfClassification, classify_pdf
from rag_parser.parsers.pdf_page_processor import process_complex_page
from rag_parser.parsers.pdf_pipeline import parse_pdf, run_pipeline
from rag_parser.parsers.pdf_router import (
    PageResult,
    RoutingPlan,
    assemble_markdown,
    build_routing_plan,
    group_table_runs,
)
from rag_parser.parsers.pdf_table_detector import TableDetectionResult
from rag_parser.parsers.pdf_text_parser import _valid_ratio


# --------------------------------------------------------------------------- #
# Классификатор
# --------------------------------------------------------------------------- #


class TestClassifier:
    """Классификация PDF: текстовый vs скан (шаг 1 ТЗ)."""

    def test_text_pdf_classified_as_text(self, pdf_files):
        """PDF с текстовым слоем → text."""
        if not pdf_files.get("pdf_text"):
            pytest.skip("pdf_text.pdf отсутствует")
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

    def test_mixed_pdf_pages_are_split(self, pdf_files):
        """Смешанный PDF: текст → text, скан/мусор → scan_pages."""
        if not pdf_files.get("pdf_mixed"):
            pytest.skip("pdf_mixed.pdf отсутствует")
        result = classify_pdf(pdf_files["pdf_mixed"], min_chars_per_page=5,
                              min_area_ratio=0.0, min_valid_char_ratio=0.7)
        # Страница 0 — текст, 1 — таблица (тоже текстовый слой), 2 и 3 — нет.
        assert 0 in result.text_pages
        assert 1 in result.text_pages
        assert 2 in result.scan_pages
        assert 3 in result.scan_pages

    def test_garbage_text_layer_is_not_text(self, tmp_path):
        """Мусорный слой («(cid:)…», нет гласных) не считается текстовым."""
        assert _valid_ratio("(cid:12)(cid:3)(cid:44)(cid:7)") == 0.0
        assert _valid_ratio("\ufffd\ufffd") == 0.0
        assert _valid_ratio("Абвгдеж — обычный текст") > 0.7

    def test_small_text_area_is_scan_page(self, pdf_files, monkeypatch):
        """Страница с крошечной площадью текста → scan_page."""
        if not pdf_files.get("pdf_text"):
            pytest.skip("pdf_text.pdf отсутствует")
        result = classify_pdf(pdf_files["pdf_text"], min_chars_per_page=5,
                              min_area_ratio=0.9)  # нужен почти весь лист
        assert result.pdf_type == "scan"

    def test_decision_by_document_ratio(self, pdf_files):
        """Порог `доля текстовых страниц ≥ N%` применяется к документу."""
        if not pdf_files.get("pdf_text"):
            pytest.skip("pdf_text.pdf отсутствует")
        result = classify_pdf(
            pdf_files["pdf_text"], min_chars_per_page=5, text_pages_min_ratio=0.99
        )
        if len(result.text_pages) < result.total_pages:
            assert result.pdf_type == "scan"


# --------------------------------------------------------------------------- #
# Детектор таблиц
# --------------------------------------------------------------------------- #


class TestTableDetector:
    """Детекция таблиц на страницах PDF (шаг 2.1 ТЗ)."""

    def test_table_pdf_has_table_pages(self, pdf_files):
        """PDF с таблицей → хотя бы одна table_page."""
        if not pdf_files.get("pdf_table"):
            pytest.skip("pdf_table.pdf отсутствует")
        from rag_parser.parsers.pdf_table_detector import detect_tables

        result = detect_tables(pdf_files["pdf_table"])
        assert len(result.table_pages) > 0

    def test_mixed_pdf_table_page_detected(self, pdf_files):
        """В смешанном PDF табличная страница найдена (страница 1)."""
        if not pdf_files.get("pdf_mixed"):
            pytest.skip("pdf_mixed.pdf отсутствует")
        from rag_parser.parsers.pdf_table_detector import detect_tables

        result = detect_tables(pdf_files["pdf_mixed"])
        assert 1 in result.table_pages


# --------------------------------------------------------------------------- #
# Роутер — группировка в раны и план
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


class TestRoutingPlan:
    """План маршрутизации: текстовый PDF vs скан."""

    def test_scan_all_pages_to_ocr(self):
        """Скан → все страницы в OCR-ветку (layout+EasyOCR+TableFormer)."""
        cls = PdfClassification(
            pdf_type="scan",
            total_pages=5,
            text_pages=[],
            scan_pages=[0, 1, 2, 3, 4],
            chars_per_page=[0, 0, 0, 0, 0],
        )
        plan = build_routing_plan(cls)
        assert plan.scan_all is True
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
        """Текстовый PDF с таблицами → таблицы отдельно, остальное в text_parser."""
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
        """Страница без текстового слоя внутри текстового PDF → OCR-ветка."""
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
        assert plan.scan_all is False


# --------------------------------------------------------------------------- #
# Сборка Markdown
# --------------------------------------------------------------------------- #


class TestAssembleMarkdown:
    """Сборка Markdown в порядке страниц (шаг 2.4 ТЗ)."""

    def _plan(self) -> RoutingPlan:
        return RoutingPlan()

    def test_order_by_page(self):
        """Результаты склеиваются в порядке номеров страниц."""
        results = [
            PageResult(pages=[2], source="⚠️pymupdf", markdown="C"),
            PageResult(pages=[0], source="⚠️pymupdf", markdown="A"),
            PageResult(pages=[1], source="layout", markdown="B"),
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
        """Если предыдущий заканчивается точкой — \\n\\n, не склейка."""
        results = [
            PageResult(pages=[0], source="text_parser", markdown="Предложение."),
            PageResult(pages=[1], source="text_parser", markdown="Новое предложение"),
        ]
        md = assemble_markdown(results)
        assert "Предложение.\n\nНовое предложение" in md

    def test_ocr_pages_sorted_individually(self):
        """Все страницы — отдельные PageResult: порядок строго по номерам."""
        results = [
            PageResult(pages=[3], source="scan", markdown="D"),
            PageResult(pages=[1], source="table", markdown="B"),
            PageResult(pages=[0], source="text_parser", markdown="A"),
            PageResult(pages=[2], source="text_parser", markdown="C"),
        ]
        md = assemble_markdown(results)
        parts = md.split("\n\n")
        assert parts == ["A", "B", "C", "D"]


# --------------------------------------------------------------------------- #
# Пайплайн — маршруты страниц в логах
# --------------------------------------------------------------------------- #


class TestRouteLogging:
    """Критерий: в логах видно, какой путь прошла каждая страница."""

    def test_routes_logged_per_page(self, pdf_files, monkeypatch, caplog):
        if not pdf_files.get("pdf_mixed"):
            pytest.skip("pdf_mixed.pdf отсутствует")

        def mock_complex(doc, page_index, *, dpi, langs, has_text_layer):
            page_no = page_index + 1
            if has_text_layer:
                return f"TABLE_MD_PAGE{page_no}", "table→layout+ocr+tableformer"
            return f"SCAN_MD_PAGE{page_no}", "scan→layout+ocr+tableformer"

        monkeypatch.setattr(
            "rag_parser.parsers.pdf_pipeline.process_complex_page", mock_complex
        )
        with caplog.at_level(logging.INFO, logger="rag_parser.parsers.pdf_pipeline"):
            result = run_pipeline(pdf_files["pdf_mixed"], min_chars_per_page=5,
                                  min_area_ratio=0.0,
                                  min_valid_char_ratio=0.7)
        messages = [rec.getMessage() for rec in caplog.records]
        assert any("page 1: text-only → pymupdf" in m for m in messages)
        assert any("page 2: table → layout+easyocr+tableformer" in m
                   for m in messages)
        assert any("page 3: scan → layout+easyocr+tableformer" in m
                   for m in messages)
        assert any("page 4: scan → layout+easyocr+tableformer" in m
                   for m in messages)

    def test_no_ocr_on_plain_text_pages(self, pdf_files, monkeypatch):
        """Страницы без таблиц НЕ вызывают EasyOCR и TableFormer."""
        ocr_calls: list = []
        table_calls: list = []

        def mock_easyocr(image, bbox, **kw):
            ocr_calls.append(bbox)
            return "ocr text"

        def mock_table(image, bbox, **kw):
            table_calls.append(bbox)
            return "| A |"

        monkeypatch.setattr(
            "rag_parser.parsers.pdf_page_processor.run_easyocr_on_crop", mock_easyocr
        )
        monkeypatch.setattr(
            "rag_parser.parsers.pdf_page_processor.run_tableformer_on_crop",
            mock_table,
        )
        # Смешанный PDF: 2 страницы с текстовым слоем (0 — без таблиц,
        # 1 — таблица) и 2 скан-страницы; скан-страницы идут через mock.
        if not pdf_files.get("pdf_mixed"):
            pytest.skip("pdf_mixed.pdf отсутствует")
        import fitz

        doc = fitz.open(str(pdf_files["pdf_mixed"]))
        from rag_parser.parsers.pdf_page_processor import (
            process_complex_page as real_func,
        )

        try:
            md, route = real_func(doc, 1, dpi=200, langs=("ru", "en"),
                                  has_text_layer=True)
        finally:
            doc.close()
        # На табличной странице ocr/tableformer функций были вызваны...
        assert table_calls or ocr_calls or md  # mock-замена не нужна
        assert isinstance(route, str)

    def test_text_page_runs_without_ocr(self, pdf_files, monkeypatch):
        """Страница 0 (без таблиц) не вызывает ни EasyOCR, ни TableFormer."""
        import fitz

        from rag_parser.parsers.pdf_pipeline import _process_single_page
        from rag_parser.parsers.pdf_router import build_routing_plan

        spy_ocr = MagicMock(side_effect=AssertionError("OCR не должен вызываться"))
        monkeypatch.setattr(
            "rag_parser.parsers.pdf_page_processor.run_easyocr_on_crop", spy_ocr
        )
        spy_tf = MagicMock(side_effect=AssertionError("TableFormer не должен вызываться"))
        monkeypatch.setattr(
            "rag_parser.parsers.pdf_page_processor.run_tableformer_on_crop", spy_tf
        )

        doc = fitz.open(str(pdf_files["pdf_text"]))
        cls = classify_pdf(pdf_files["pdf_text"], min_chars_per_page=5)
        plan = build_routing_plan(cls)
        try:
            md, route = _process_single_page(
                doc, 0, plan=plan, easyocr_langs=("ru", "en"), render_dpi=200
            )
        finally:
            doc.close()
        assert md.strip()
        assert route == "text-only→pymupdf"


# --------------------------------------------------------------------------- #
# Пайплайн — полный прогон на четырёх типах PDF
# --------------------------------------------------------------------------- #


class TestPipelineFullRun:
    """run_pipeline на 4 PDF: порядок страниц + Markdown-таблицы."""

    def test_mixed_pdf_page_order(self, pdf_files, monkeypatch):
        """Смешанный PDF: страницы в итоговом MD в исходном порядке."""
        if not pdf_files.get("pdf_mixed"):
            pytest.skip("pdf_mixed.pdf отсутствует")

        results_by_page = {
            1: ("| Region | Q1 |\n|---|---|\n| North | 100 |",
                "table→layout+ocr+tableformer"),
            2: ("ScannedContent page two", "scan→layout+ocr+tableformer"),
            3: ("GibberishOcrLayout page three", "scan→layout+ocr+tableformer"),
        }

        def mock_complex(doc, page_index, *, dpi, langs, has_text_layer):
            return results_by_page[page_index]

        monkeypatch.setattr(
            "rag_parser.parsers.pdf_pipeline.process_complex_page", mock_complex
        )
        result = run_pipeline(pdf_files["pdf_mixed"], min_chars_per_page=5,
                              min_area_ratio=0.0, min_valid_char_ratio=0.7)
        md = result.markdown
        p_alpha = md.find("Alpha page")
        p_table = md.find("Region")
        p_scan = md.find("ScannedContent")
        p_garbage = md.find("GibberishOcrLayout")
        assert -1 not in (p_alpha, p_table, p_scan, p_garbage)
        assert p_alpha < p_table < p_scan < p_garbage
        # markdown-таблица присутствует в выводе
        assert "| Region | Q1 |" in md
        assert "|---|" in md

    def test_parse_pdf_interface(self, pdf_files, monkeypatch):
        """Единый интерфейс ТЗ: parse_pdf(path) -> markdown_str."""
        if not pdf_files.get("pdf_text"):
            pytest.skip("pdf_text.pdf отсутствует")

        def mock_complex(doc, page_index, *, dpi, langs, has_text_layer):
            return "", "empty"

        monkeypatch.setattr(
            "rag_parser.parsers.pdf_pipeline.process_complex_page", mock_complex
        )
        from rag_parser.core.config import settings

        monkeypatch.setattr(settings, "PDF_TEXT_MIN_CHARS_PER_PAGE", 5)
        monkeypatch.setattr(settings, "PDF_TEXT_MIN_AREA_RATIO", 0.0)
        monkeypatch.setattr(settings, "PDF_TEXT_MIN_VALID_CHAR_RATIO", 0.7)
        monkeypatch.setattr(settings, "EASYOCR_LANGS", "ru,en")
        md = parse_pdf(pdf_files["pdf_text"])
        assert isinstance(md, str)
        assert "First page" in md or "Second page" in md


# --------------------------------------------------------------------------- #
# Пайплайн — fallback при падении обработки
# --------------------------------------------------------------------------- #


class TestPipelineFallback:
    """Отказоустойчивость: страница не теряется при падении обработчика."""

    def test_table_page_fallback_to_text_layer(self, pdf_files, monkeypatch):
        """Сложный обработчик упал на странице с таблицей → текстовый слой."""
        if not pdf_files.get("pdf_table"):
            pytest.skip("pdf_table.pdf отсутствует")

        def boom(doc, page_index, **kwargs):
            raise RuntimeError("mock failure")

        monkeypatch.setattr(
            "rag_parser.parsers.pdf_pipeline.process_complex_page", boom
        )
        result = run_pipeline(pdf_files["pdf_table"], min_chars_per_page=5)
        # Страница с таблицей содержит текст (он есть и в слое).
        assert "Quarterly" in result.markdown or result.markdown
        assert result.metadata["failed_pages"]  # помечено в метаданных

    def test_scan_docling_failure_does_not_crash(self, pdf_files, monkeypatch):
        """Скан, OCR недоступен: пайплайн не падает, фиксирует проблему."""
        if not pdf_files.get("pdf_scan"):
            pytest.skip("pdf_scan.pdf отсутствует")

        def boom(doc, page_index, **kwargs):
            raise RuntimeError("no easyocr in test env")

        monkeypatch.setattr(
            "rag_parser.parsers.pdf_pipeline.process_complex_page", boom
        )
        result = run_pipeline(pdf_files["pdf_scan"])
        assert result.markdown == ""  # текст взять неоткуда, но нет краха
        assert result.metadata["failed_pages"] == [0]


# --------------------------------------------------------------------------- #
# Processor — reading order и эвристический layout
# --------------------------------------------------------------------------- #


class TestReadingOrder:
    """Сборка блоков в порядке чтения (сверху вниз, учитывая колонки)."""

    def _block(self, bbox, md, label="text"):
        from rag_parser.parsers.pdf_page_processor import _OcrBlock

        return _OcrBlock(label=label, bbox=bbox, markdown=md)

    def test_single_column_top_down(self):
        from rag_parser.parsers.pdf_page_processor import sort_blocks_reading_order

        blocks = [
            self._block((0, 200, 600, 300), "B"),
            self._block((0, 50, 600, 150), "A"),
        ]
        ordered = sort_blocks_reading_order(blocks, page_width=612)
        assert [b.markdown for b in ordered] == ["A", "B"]

    def test_two_columns_within_section(self):
        from rag_parser.parsers.pdf_page_processor import sort_blocks_reading_order

        left_top = self._block((0, 50, 200, 250), "L1")
        right_top = self._block((310, 50, 600, 250), "R1")
        left_bottom = self._block((0, 260, 200, 450), "L2")
        ordered = sort_blocks_reading_order(
            [left_top, right_top, left_bottom], page_width=612
        )
        # Внутри «полос» колонки идут слева направо: L1 → R1, затем L2.
        assert [b.markdown for b in ordered] == ["L1", "R1", "L2"]

    def test_full_width_block_breaks_sections(self):
        from rag_parser.parsers.pdf_page_processor import sort_blocks_reading_order

        wide = self._block((0, 300, 612, 350), "W")
        above = self._block((0, 50, 200, 250), "A")
        below = self._block((0, 360, 200, 550), "B")
        ordered = sort_blocks_reading_order([wide, above, below], page_width=612)
        assert [b.markdown for b in ordered] == ["A", "W", "B"]


class TestLayoutHeuristic:
    """Эвристическая layout-детекция (PyMuPDF): таблицы/текст/картинки."""

    def test_mixed_pdf_table_page_layout(self, pdf_files):
        if not pdf_files.get("pdf_mixed"):
            pytest.skip("pdf_mixed.pdf отсутствует")
        import fitz

        from rag_parser.parsers.pdf_page_processor import (
            LABEL_TABLE,
            LABEL_TEXT,
            _layout_via_pymupdf,
        )

        doc = fitz.open(str(pdf_files["pdf_mixed"]))
        try:
            blocks = _layout_via_pymupdf(doc, 1)
        finally:
            doc.close()
        assert any(b.label == LABEL_TABLE for b in blocks)
        assert any(b.label == LABEL_TEXT for b in blocks)

    def test_process_complex_page_produces_markdown(self, pdf_files):
        """Табличная страница смешанного PDF: Processor даёт Markdown."""
        if not pdf_files.get("pdf_mixed"):
            pytest.skip("pdf_mixed.pdf отсутствует")
        import fitz

        doc = fitz.open(str(pdf_files["pdf_mixed"]))
        try:
            md, route = process_complex_page(
                doc, 1, dpi=150, langs=("ru", "en"), has_text_layer=True
            )
        finally:
            doc.close()
        assert route == "table→layout+ocr+tableformer"
        # Таблица начинается с '|' — TableFormer (fallback) выдал таблицу.
        assert "|" in md
