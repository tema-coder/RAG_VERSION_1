"""Тесты маршрута PDF → DOCX (parsers/pdf_via_docx).

Сам конвертёр не проверяем — он внешний. Проверяем маршрутизацию: флаг
включён по умолчанию, файл реально идёт через DOCX, временный файл
удаляется, а артефакты конвертации (таблица из одной строки шапки) не
попадают в итоговое дерево.
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

    def test_flag_is_on_by_default(self):
        """Эталон (раздел 4.2 спеки): по умолчанию PDF идёт через
        pdf2docx → DocxReader, а не через нативный PdfTxtlayerReader."""
        from rag_parser.core.config import Settings

        assert Settings().PDF_VIA_DOCX is True

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
    """Проблема 3: конвертер ``pdf2docx`` работает приблизительно.

    Замер (тесты ниже + ``scripts/compare_all_formats.py``)::

        показатель         эталонный DOCX    PDF -> DOCX
        заголовков                        42            37
        таблиц                              9             9
        orphan                              0             1

    Часть текста теряется, слова склеиваются без пробелов, а таблица,
    разорванная границей страницы, даёт хвост без строки шапки — он и
    остаётся единственной orphan-таблицей. Куски из одной строки шапки
    отбрасываются на входе (см. ``dedoc_tree.drop_header_only_tables``),
    поэтому в дереве таблиц ровно столько же, сколько в эталоне.
    Тесты фиксируют фактическое поведение, чтобы расхождение с критерием
    ТЗ было видно в коде, а не только в переписке.
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
        """Сырой конвертер добавляет лишнюю таблицу — их больше, чем в
        оригинале (её разбирает фильтр ниже)."""
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
    def test_pdf_route_drops_header_only_tables(self, monkeypatch, tmp_path):
        """В дереве PDF столько же таблиц, сколько в эталонном DOCX.

        Конвертер оставляет куски из одной строки шапки на месте стыка
        страниц — это не таблицы документа, и ``drop_header_only_tables``
        отбрасывает их до привязки.
        """
        from rag_parser.parsers import dedoc_parser as module
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
        raw_tables = count_tables(DedocParser().parse(converted_path).tree)

        monkeypatch.setattr(module.settings, "PDF_VIA_DOCX", True)
        monkeypatch.setattr(module.settings, "TEMP_DIR", str(tmp_path), raising=False)
        pdf_tables = count_tables(DedocParser().parse(pdf).tree)
        docx_tables = count_tables(DedocParser().parse(docx).tree)

        assert raw_tables > docx_tables, "сырой конвертер добавляет куски таблиц"
        assert pdf_tables == docx_tables

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

    def test_pdf_via_docx_is_the_default(self):
        """Эталон (раздел 4.2 спеки): по умолчанию идёт маршрут через
        pdf2docx → DocxReader, а не нативный PdfTxtlayerReader."""
        import rag_parser.parsers.dedoc_parser as module

        assert module.settings.PDF_VIA_DOCX is True
