"""Тесты опционального маршрута PDF → DOCX (parsers/pdf_via_docx).

Сам конвертёр не проверяем — он внешний. Проверяем маршрутизацию: флаг
выключен по умолчанию, при включении файл реально идёт через DOCX, а
временный файл удаляется.
"""

from pathlib import Path

import pytest

from rag_parser.parsers.pdf_via_docx import convert_pdf_to_docx, pdf2docx_available

INPUT = Path(__file__).resolve().parent.parent.parent / "input"


class TestAvailability:
    def test_import_works(self):
        """Функция обязана возвращать bool, а не падать."""
        assert isinstance(pdf2docx_available(), bool)


class TestConvert:
    @pytest.mark.skipif(
        not pdf2docx_available(), reason="pdf2docx не установлен"
    )
    def test_converts_real_pdf(self, tmp_path):
        source = INPUT / "gost_report.pdf"
        if not source.exists():
            pytest.skip("файл отсутствует в input/")

        result = convert_pdf_to_docx(source, tmp_path)

        assert result is not None and result.exists()
        assert result.suffix == ".docx"
        assert result.stat().st_size > 0

    def test_missing_file_returns_none(self, tmp_path):
        assert convert_pdf_to_docx(tmp_path / "нет.pdf", tmp_path) is None

    @pytest.mark.skipif(not pdf2docx_available(), reason="pdf2docx не установлен")
    def test_broken_pdf_returns_none(self, tmp_path):
        """Битый PDF не должен ронять пайплайн."""
        source = tmp_path / "broken.pdf"
        source.write_bytes(b"%PDF-1.4\nnot really a pdf")
        assert convert_pdf_to_docx(source, tmp_path) is None


class TestParserRouting:
    """Куда реально уходит файл: в PdfReader или в DocxReader."""

    def _record_routes(self, monkeypatch):
        """Подменяет выбор менеджера, запоминая расширения, но разбор
        оставляет настоящий — иначе тест ничего не проверяет."""
        import rag_parser.parsers.dedoc_parser as module

        real = module.prepare_manager_for_format
        seen = []

        def spy(extension):
            seen.append(str(extension).lower())
            return real(extension)

        monkeypatch.setattr(module, "prepare_manager_for_format", spy)
        return seen

    def test_flag_is_off_by_default(self):
        from rag_parser.core.config import Settings

        assert Settings().PDF_VIA_DOCX is False

    def test_native_pdf_route_used_when_flag_off(self, monkeypatch):
        import rag_parser.parsers.dedoc_parser as module

        monkeypatch.setattr(module.settings, "PDF_VIA_DOCX", False)
        seen = self._record_routes(monkeypatch)

        source = INPUT / "gost_report.pdf"
        if not source.exists():
            pytest.skip("файл отсутствует в input/")

        result = module.DedocParser().parse(source)
        assert result.markdown.strip()
        assert ".pdf" in seen
        assert ".docx" not in seen

    @pytest.mark.skipif(
        not pdf2docx_available(), reason="pdf2docx не установлен"
    )
    def test_docx_route_used_when_flag_on(self, monkeypatch, tmp_path):
        import rag_parser.parsers.dedoc_parser as module

        monkeypatch.setattr(module.settings, "PDF_VIA_DOCX", True)
        monkeypatch.setattr(module.settings, "TEMP_DIR", str(tmp_path), raising=False)
        seen = self._record_routes(monkeypatch)

        source = INPUT / "gost_report.pdf"
        if not source.exists():
            pytest.skip("файл отсутствует в input/")

        result = module.DedocParser().parse(source)
        assert result.markdown.strip()
        assert ".docx" in seen
        # временный DOCX не должен остаться в temp
        assert not list(tmp_path.glob("*_via_pdf2docx.docx"))

class TestPdfViaDocxMeasuredReality:
    """Проблема 3: критерий «9 таблиц, 0 orphan» измерением НЕ подтверждается.

    ТЗ предлагает проверять, что маршрут даёт 9 таблиц без orphan. Замер
    (``scripts/debug_pdf_via_docx.py``) говорит обратное:

    ==========================  =========  =========
    показатель                  DOCX       DOCX из PDF
    ==========================  =========  =========
    текстовых блоков            66         24   (-64%)
    таблиц                     9          10   (+1 лишняя)
    orphan                      0          1
    ==========================  =========  =========

    Конвертер ``pdf2docx`` восстанавливает таблицы приблизительно
    (floating-блоки по эвристическим координатам) и теряет большую часть
    текста, поэтому маршрут остаётся выключенным по умолчанию и описан
    как ограничение (README, п. 6 Workarounds). Тесты ниже фиксируют
    фактическое поведение, чтобы расхождение с критерием ТЗ было видно
    в коде, а не только в переписке.
    """

    @pytest.mark.skipif(
        not pdf2docx_available(), reason="pdf2docx не установлен"
    )
    def test_conversion_is_lossy(self, tmp_path):
        """Текст теряется: конвертированный DOCX короче исходного."""
        from rag_parser.parsers.dedoc_parser import DedocParser
        from rag_parser.parsers.pdf_via_docx import convert_pdf_to_docx

        pdf = INPUT / "gost_report.pdf"
        docx = INPUT / "gost_report.docx"
        if not pdf.exists() or not docx.exists():
            pytest.skip("исходники отсутствуют в input/")

        converted_path = convert_pdf_to_docx(pdf, tmp_path)
        assert converted_path is not None
        original = DedocParser().parse(docx)
        converted = DedocParser().parse(converted_path)
        assert len(converted.markdown) < len(original.markdown)

    def test_converted_docx_has_extra_table(self, tmp_path):
        """Конвертер добавляет лишнюю таблицу — их больше, чем в оригинале."""
        from rag_parser.parsers.dedoc_parser import DedocParser
        from rag_parser.parsers.pdf_via_docx import convert_pdf_to_docx

        def count_tables(tree):
            found = 0
            stack = [tree]
            while stack:
                node = stack.pop()
                if node.type == "table":
                    found += 1
                stack.extend(node.children)
            return found

        pdf = INPUT / "gost_report.pdf"
        docx = INPUT / "gost_report.docx"
        if not pdf.exists() or not docx.exists():
            pytest.skip("исходники отсутствуют в input/")

        converted_path = convert_pdf_to_docx(pdf, tmp_path)
        assert converted_path is not None
        original = DedocParser().parse(docx).tree
        converted = DedocParser().parse(converted_path).tree
        assert count_tables(converted) > count_tables(original)
        assert converted.metadata.get("orphan_tables", 0) >= 1

    @pytest.mark.skipif(
        not pdf2docx_available(), reason="pdf2docx не установлен"
    )
    def test_route_uses_docx_reader(self, monkeypatch, tmp_path):
        """Проверяем ровно то, что просит ТЗ: расширение меняется на .docx,
        и читает документ DocxReader, а не PdfTxtlayerReader."""
        import rag_parser.parsers.dedoc_parser as module

        monkeypatch.setattr(module.settings, "PDF_VIA_DOCX", True)
        monkeypatch.setattr(module.settings, "TEMP_DIR", str(tmp_path), raising=False)
        seen = []
        real = module.prepare_manager_for_format

        def spy(extension):
            seen.append(str(extension).lower())
            return real(extension)

        monkeypatch.setattr(module, "prepare_manager_for_format", spy)

        pdf = INPUT / "gost_report.pdf"
        if not pdf.exists():
            pytest.skip("файл отсутствует в input/")

        module.DedocParser().parse(pdf)
        assert ".docx" in seen
        assert ".pdf" not in seen

    def test_native_pdf_is_the_default(self):
        """Без флага идёт нативный PdfReader — он и даёт корректные таблицы."""
        import rag_parser.parsers.dedoc_parser as module

        assert module.settings.PDF_VIA_DOCX is False
