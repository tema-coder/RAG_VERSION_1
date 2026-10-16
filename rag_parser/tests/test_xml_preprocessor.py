"""Тесты препроцессинга XML → HTML (parsers/xml_preprocessor)."""

from pathlib import Path

import pytest

from rag_parser.parsers.xml_preprocessor import (
    prepare_xml_for_dedoc,
    xml_to_html,
)

GOST_FRAGMENT = """\
<?xml version='1.0' encoding='utf-8'?>
<gost-standard number="ГОСТ Р 99999—2026">
  <title>Синтетические документы</title>
  <preface>
    <item number="1">ПОДГОТОВЛЕН Национальной ассоциацией</item>
  </preface>
  <toc>
    <entry number="1" page="1">Область применения</entry>
  </toc>
  <introduction>
    <paragraph>Вводный текст.</paragraph>
  </introduction>
  <sections>
    <section number="1" title="Область применения">
      <paragraph>Текст раздела 1.</paragraph>
      <table rows="2" cols="2">
        <thead>
          <tr><th>Показатель</th><th>Q1</th></tr>
        </thead>
        <tbody>
          <tr><td>Выручка</td><td>100</td></tr>
        </tbody>
      </table>
      <section number="1.1" title="Подраздел">
        <paragraph>Текст подраздела.</paragraph>
        <definition number="1.1.1" term="термин">Определение термина</definition>
      </section>
    </section>
  </sections>
  <appendices>
    <appendix letter="А" type="справочное" title="Пример использования">
      <paragraph>Текст приложения.</paragraph>
    </appendix>
  </appendices>
  <bibliography>
    <reference number="[1]">ISO/IEC 19944:2017</reference>
  </bibliography>
</gost-standard>
"""

INPUT = Path(__file__).resolve().parent.parent.parent / "input"


class TestXmlToHtml:
    def test_title_becomes_h1(self):
        assert "<h1>Синтетические документы</h1>" in xml_to_html(GOST_FRAGMENT)

    def test_implicit_section_titles(self):
        html = xml_to_html(GOST_FRAGMENT)
        assert "<h2>Предисловие</h2>" in html
        assert "<h2>Содержание</h2>" in html
        assert "<h2>Введение</h2>" in html
        assert "<h2>Библиография</h2>" in html

    def test_section_number_and_title(self):
        assert "<h2>1 Область применения</h2>" in xml_to_html(GOST_FRAGMENT)

    def test_nesting_controls_level(self):
        """Вложенность section задаёт уровень, а не точки в номере."""
        html = xml_to_html(GOST_FRAGMENT)
        assert "<h3>1.1 Подраздел</h3>" in html

    def test_appendix_heading(self):
        """Метка приложения и его название — два заголовка.

        В эталонном DOCX «Приложение А (справочное)» — заголовок уровня 1,
        а «Пример использования» вложен в него заголовком уровня 2. Слитый
        заголовок «Приложение А (справочное) Пример использования» терял
        подраздел.
        """
        html = xml_to_html(GOST_FRAGMENT)
        assert "<h2>Приложение А (справочное)</h2>" in html
        assert "<h3>Пример использования</h3>" in html

    def test_table_becomes_html_table(self):
        html = xml_to_html(GOST_FRAGMENT)
        assert "<table>" in html
        assert "<th>Показатель</th>" in html
        assert "<td>Выручка</td>" in html

    def test_preface_items_become_headers(self):
        """Пункт предисловия — заголовок, как в эталонном DOCX.

        Раньше он оставался абзацем, и весь предисловие уезжало в чанки с
        ``parent_id`` предыдущего заголовка.
        """
        html = xml_to_html(GOST_FRAGMENT)
        assert "<h1>1 ПОДГОТОВЛЕН Национальной ассоциацией</h1>" in html

    def test_plain_items_stay_paragraphs(self):
        """``<item>`` вне предисловия (перечень ссылок, сокращений) — абзац."""
        html = xml_to_html(
            "<doc><section number='1' title='Р'>"
            "<item>ГОСТ Р 70000 —первый</item></section></doc>"
        )
        assert "<p>ГОСТ Р 70000 —первый</p>" in html
        assert "<h1>ГОСТ Р 70000 —первый</h1>" not in html

    def test_definition_becomes_heading_with_term_and_text(self):
        """Термин — заголовок: номер, термин и сам текст определения."""
        html = xml_to_html(GOST_FRAGMENT)
        assert "<h4>1.1.1 термин: Определение термина</h4>" in html

    def test_toc_entry_gets_page_marker(self):
        """Строка оглавления несёт номер страницы — иначе она не гасится."""
        html = xml_to_html(GOST_FRAGMENT)
        assert "<p>1 Область применения … 1</p>" in html

    def test_standard_number_becomes_heading(self):
        """Обозначение стандарта из атрибута корня — заголовок титула."""
        assert "<h1>ГОСТ Р 99999—2026</h1>" in xml_to_html(GOST_FRAGMENT)

    def test_reference_keeps_number(self):
        assert "<p>[1] ISO/IEC 19944:2017</p>" in xml_to_html(GOST_FRAGMENT)

    def test_unknown_tags_keep_their_text(self):
        """Неизвестная схема не приводит к молчаливой потере текста."""
        html = xml_to_html("<doc><mystery>Важный текст</mystery></doc>")
        assert "Важный текст" in html

    def test_special_chars_are_escaped(self):
        html = xml_to_html("<doc><paragraph>A &amp; B &lt;tag&gt;</paragraph></doc>")
        assert "<tag>" not in html
        assert "&amp;" in html and "&lt;tag&gt;" in html

    def test_invalid_xml_returns_empty(self):
        assert xml_to_html("<doc><unclosed>") == ""

    def test_empty_document(self):
        assert xml_to_html("<doc/>") == "<html><body></body></html>"

    def test_namespace_is_stripped(self):
        html = xml_to_html(
            '<d:doc xmlns:d="urn:x"><d:section number="1" title="Раздел"/>'
            "<d:paragraph>Текст</d:paragraph></d:doc>"
        )
        assert "<h2>1 Раздел</h2>" in html
        assert "<p>Текст</p>" in html


class TestPrepareXmlForDedoc:
    def test_writes_html_with_markers(self, tmp_path):
        source = tmp_path / "doc.xml"
        source.write_text(GOST_FRAGMENT, encoding="utf-8")

        temp_path, tables = prepare_xml_for_dedoc(source, tmp_path / "out")

        assert temp_path is not None and temp_path.suffix == ".html"
        assert len(tables) == 1
        html = temp_path.read_text(encoding="utf-8")
        assert "<table>" not in html
        assert "__RAGTABLE_0__" in html

    def test_invalid_xml_is_reported(self, tmp_path):
        source = tmp_path / "bad.xml"
        source.write_text("<doc><unclosed>", encoding="utf-8")
        temp_path, tables = prepare_xml_for_dedoc(source, tmp_path / "out")
        assert temp_path is None and tables == []

    def test_document_without_tables(self, tmp_path):
        source = tmp_path / "plain.xml"
        source.write_text("<doc><section number='1' title='Р'/></doc>", encoding="utf-8")
        temp_path, tables = prepare_xml_for_dedoc(source, tmp_path / "out")
        assert tables == []
        assert "<h2>1 Р</h2>" in temp_path.read_text(encoding="utf-8")


class TestXmlEndToEnd:
    """Сквозная проверка на реальном .xml из input/."""

    def test_real_xml_tables_and_structure(self):
        from rag_parser.parsers.dedoc_parser import DedocParser
        from rag_parser.parsers.dedoc_tree import collect_headers

        source = INPUT / "gost_report.xml"
        if not source.exists():
            pytest.skip("файл отсутствует в input/")

        result = DedocParser().parse(source)
        assert result.tree.metadata["tables_by_marker"] == 9
        assert result.tree.metadata["orphan_tables"] == 0

        headers = collect_headers(result.tree)
        levels = {h.level for h in headers}
        # Вложенность 1 -> 6.2.1 восстановлена (после сдвига уровней под
        # эталонную шкалу: тело документа в XML начинается с <h2>).
        assert {1, 2, 3} <= levels
        titles = [h.text for h in headers]
        assert any("Область применения" in t for t in titles)
        assert any(t.startswith("Приложение") for t in titles)

    def test_preface_items_are_headers_of_level_one(self):
        """Пункты предисловия — заголовки уровня 1 (как в эталонном DOCX)."""
        from rag_parser.parsers.dedoc_parser import DedocParser
        from rag_parser.parsers.dedoc_tree import collect_headers

        source = INPUT / "gost_report.xml"
        if not source.exists():
            pytest.skip("файл отсутствует в input/")

        result = DedocParser().parse(source)
        headers = collect_headers(result.tree)
        preface = [h for h in headers if h.text.startswith("1 ПОДГОТОВЛЕН")]
        assert len(preface) == 1
        assert preface[0].level == 1

    def test_toc_lines_are_suppressed(self):
        """Строки оглавления гасятся, а не становятся заголовками."""
        from rag_parser.parsers.dedoc_parser import DedocParser
        from rag_parser.parsers.dedoc_tree import collect_headers

        source = INPUT / "gost_report.xml"
        if not source.exists():
            pytest.skip("файл отсутствует в input/")

        result = DedocParser().parse(source)
        assert result.tree.metadata["toc_suppressed"] == 19
        titles = [h.text for h in collect_headers(result.tree)]
        assert not any("Область применения … 1" in t for t in titles)

    def test_no_marker_leaks_into_markdown(self):
        from rag_parser.parsers.dedoc_parser import DedocParser

        source = INPUT / "gost_report.xml"
        if not source.exists():
            pytest.skip("файл отсутствует в input/")

        result = DedocParser().parse(source)
        assert "__RAGTABLE_" not in result.markdown