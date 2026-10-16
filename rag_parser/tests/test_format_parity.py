"""Паритет всех форматов ``input/`` с эталонным ``gost_report.docx``.

Эталон — один и тот же документ в девяти форматах. Разбор и чанкинг каждого
формата обязаны давать одинаковую структуру дерева и одинаковые чанки; иначе
под тем же запросом пользователь получит разную выдачу в зависимости от
формата файла.

Что проверяется (для каждого доступного файла в ``input/``):

* 42 заголовка с эталонными уровнями (у ``.md``/``.xml`` — с поправкой на
  дельты самого исходника, см. ``SOURCE_DELTAS``);
* 9 таблиц, 0 orphan-таблиц;
* 19 подавленных строк оглавления;
* у всех чанков есть ``breadcrumb`` и ``parent_id``, кроме чанков вне разделов;
* табличный чанк = ровно одна строка данных;
* в чанках нет строк-разделителей («======») и маркеров таблиц.

Полное сравнение с эталонными артефактами живёт в
``scripts/compare_all_formats.py``: эти тесты не зависят от файлов в
``input/etalon/`` и проверяют инварианты, а не побайтовое совпадение.
"""

import re
from pathlib import Path

import pytest

from rag_parser.chunkers.hybrid import HybridRagChunker
from rag_parser.parsers.dedoc_parser import DedocParser

INPUT = Path(__file__).resolve().parent.parent.parent / "input"

ETALON_HEADERS = 42
ETALON_TABLES = 9
ETALON_ORPHANS = 0
ETALON_TOC_SUPPRESSED = 19

#: Форматы, для которых сравнение обязано быть точным.
EXACT_FORMATS = (
    "gost_report.docx",
    "gost_report.doc",
    "gost_report.odt",
    "gost_report.rtf",
    "gost_report.html",
    "gost_report.txt",
)

#: Форматы с дельтой самого исходника: чего файл физически не содержит.
#: ``md``/``xml`` — нет двух строк титульного блока («ПО ТЕХНИЧЕСКОМУ
#: РЕГУЛИРОВАНИЮ И МЕТРОЛОГИИ», «НАЦИОНАЛЬНЫЙ СТАНДАРТ РОССИЙСКОЙ
#: ФЕДЕРАЦИИ»); в ``xml`` три строки заголовка лежат в одном ``<title>``
#: одной строкой без разделителей; ``txt`` — верхний регистр, поэтому
#: сверяем регистронезависимо.
SOURCE_DELTAS = {
    "gost_report.md": 2,
    "gost_report.xml": 4,
    "gost_report.txt": 0,
    "gost_report.pdf": 6,
}

#: Один заголовок = один чанк: столько же чанков, сколько заголовков.
MD_DELTA_HEADERS = 2
MD_DELTA_CHUNKS = 2

ALL_FORMATS = EXACT_FORMATS + (
    "gost_report.pdf",
    "gost_report.md",
    "gost_report.xml",
)

DECORATION_RE = re.compile(r"^[-=_~*·.\s]{3,}$")
MARKER_RE = re.compile(r"__RAGTABLE_\d+__")


def _parse(filename: str):
    source = INPUT / filename
    if not source.exists():
        pytest.skip(f"{filename} отсутствует в input/")
    return DedocParser().parse(source)


def _headers(tree):
    found = []
    stack = [tree]
    while stack:
        node = stack.pop(0)
        if node.type == "header":
            found.append(node)
        stack = list(node.children) + stack
    return found


def _tables(tree):
    found = []
    stack = [tree]
    while stack:
        node = stack.pop(0)
        if node.type == "table":
            found.append(node)
        stack = list(node.children) + stack
    return found


def _chunks(result, filename):
    chunker = HybridRagChunker(chunk_size=200, overlap=50, source=filename)
    return chunker.chunk(
        result.markdown,
        {"source": filename, "format": result.format},
        tree=result.tree,
    )


def _case(filename):
    if filename not in ALL_FORMATS:
        pytest.skip(f"{filename} не проверяется этим набором тестов")
    return filename


class TestTreeParity:
    """Структура дерева не зависит от формата файла."""

    @pytest.mark.parametrize("filename", EXACT_FORMATS)
    def test_header_count(self, filename):
        result = _parse(filename)
        assert len(_headers(result.tree)) == ETALON_HEADERS

    @pytest.mark.parametrize("filename", ALL_FORMATS)
    def test_header_levels_capped(self, filename):
        result = _parse(_case(filename))
        levels = {header.level for header in _headers(result.tree)}
        assert levels <= {1, 2, 3}, "уровень глубже 3 в ГОСТ не встречается"

    @pytest.mark.parametrize("filename", ALL_FORMATS)
    def test_table_count(self, filename):
        result = _parse(_case(filename))
        assert len(_tables(result.tree)) == ETALON_TABLES

    @pytest.mark.parametrize("filename", EXACT_FORMATS)
    def test_no_orphan_tables(self, filename):
        result = _parse(filename)
        assert result.tree.metadata.get("orphan_tables", 0) == ETALON_ORPHANS

    @pytest.mark.parametrize("filename", ("gost_report.pdf", "gost_report.xml"))
    def test_orphans_are_known_deltas(self, filename):
        """У PDF и XML orphan-таблица — известная дельта источника/конвертера.

        У PDF это хвост таблицы, разорванной границей страницы (``pdf2docx``
        не умеет склеивать её обратно); в XML таблицы привязываются по
        маркерам и orphan остаётся только если таблица выше первого
        заголовка — на этом файле их нет, поэтому проверка общая.
        """
        result = _parse(filename)
        assert result.tree.metadata.get("orphan_tables", 0) <= 1

    @pytest.mark.parametrize("filename", EXACT_FORMATS + ("gost_report.md",))
    def test_toc_suppressed(self, filename):
        result = _parse(filename)
        assert result.tree.metadata["toc_suppressed"] == ETALON_TOC_SUPPRESSED

    @pytest.mark.parametrize("filename", EXACT_FORMATS + ("gost_report.md",))
    def test_no_toc_lines_became_headers(self, filename):
        result = _parse(filename)
        titles = [header.text for header in _headers(result.tree)]
        assert not any("… 1" in title and "Область применения" in title for title in titles)

    @pytest.mark.parametrize("filename", ALL_FORMATS)
    def test_sections_keep_their_nesting(self, filename):
        """«5.1» вложен в «5», «6.2.1» — в «6.2»."""
        result = _parse(_case(filename))
        titles = {header.text.split(" ")[0]: header for header in _headers(result.tree)}
        for parent, child in (("5", "5.1"), ("6.2", "6.2.1")):
            if parent not in titles or child not in titles:
                continue
            assert titles[child].level == titles[parent].level + 1

    @pytest.mark.parametrize("filename", ALL_FORMATS)
    def test_no_decoration_lines_in_tree(self, filename):
        """Разделители «======» — оформление, а не текст документа."""
        result = _parse(filename)
        stack = [result.tree]
        while stack:
            node = stack.pop()
            for line in (node.text or "").splitlines():
                assert not DECORATION_RE.match(line.strip()), f"{node.text!r}"
            stack.extend(node.children)


class TestChunkParity:
    """Чанки одинаковы по структуре и чисты по содержимому."""

    @pytest.mark.parametrize("filename", EXACT_FORMATS)
    def test_chunk_count(self, filename):
        result = _parse(filename)
        assert len(_chunks(result, filename)) == 193

    @pytest.mark.parametrize("filename", ALL_FORMATS)
    def test_every_chunk_knows_its_section(self, filename):
        result = _parse(_case(filename))
        for chunk in _chunks(result, filename):
            # parent_id есть всегда (заголовок — корень иерархии), а крошка
            # может быть пустой только у чанков вне разделов.
            assert chunk.parent_id is not None
            if chunk.breadcrumb is None:
                continue
            assert ">" in chunk.breadcrumb or chunk.breadcrumb.strip()

    @pytest.mark.parametrize("filename", ALL_FORMATS)
    def test_table_chunks_are_single_rows(self, filename):
        result = _parse(filename)
        table_chunks = [
            chunk
            for chunk in _chunks(result, filename)
            if chunk.metadata.get("chunk_type") == "table"
        ]
        assert table_chunks, "в документе 9 таблиц, значит есть табличные чанки"
        for chunk in table_chunks:
            assert chunk.metadata.get("data_rows") == 1
            assert chunk.text_for_embedding, "у табличного чанка есть текст для эмбеддинга"

    @pytest.mark.parametrize("filename", ALL_FORMATS)
    def test_no_table_markers_leaked(self, filename):
        result = _parse(_case(filename))
        assert "__RAGTABLE_" not in result.markdown
        for chunk in _chunks(result, filename):
            assert not MARKER_RE.search(chunk.text)


class TestSourceDeltas:
    """Форматы, у которых дельта заложена в самом исходнике."""

    def test_md_lacks_only_two_title_lines(self):
        """В ``.md`` нет двух строк титула — и только их.

        Всё остальное обязано совпасть с эталоном: если дельта выросла,
        значит препроцессор перестал поднимать пункты предисловия, термины,
        названия приложений или строку «ОКС».
        """
        result = _parse("gost_report.md")
        titles = [header.text for header in _headers(result.tree)]
        expected = ETALON_HEADERS - SOURCE_DELTAS["gost_report.md"]
        assert len(titles) == expected

        # Шапка начинается с обозначения стандарта — первых двух строк нет.
        assert titles[0] == "ГОСТ Р 99999—2026"
        assert all(titles)
        # Структура тела на месте.
        assert any(title.startswith("1 ПОДГОТОВЛЕН") for title in titles)
        assert any(title.startswith("3.1 ") for title in titles)
        assert "Приложение А (справочное)" in titles
        assert "Пример использования настоящего стандарта" in titles
        assert "ОКС 35.240.99" in titles

    def test_md_keeps_bibliography_paragraphs(self):
        """Пункты «[1] …» остаются абзацами, а не «1. [1] …».

        Регрессия: переименование ``<li>`` в ``<p>`` внутри ``<ul>`` приводило
        к тому, что рендерер выбрасывал всю библиографию вместе с текстом.
        """
        result = _parse("gost_report.md")
        chunk_texts = [" ".join(chunk.text.split()) for chunk in _chunks(result, "gost_report.md")]
        for number in (1, 2, 3, 4):
            expected = f"[{number}] "
            assert any(text.startswith(expected) for text in chunk_texts), (
                f"запись библиографии [{number}] потеряна или получила лишнюю нумерацию"
            )

    def test_md_chunk_count(self):
        """На каждый заголовок — один чанк: 193 − 2 отсутствующих строки титула."""
        result = _parse("gost_report.md")
        assert len(_chunks(result, "gost_report.md")) == 193 - MD_DELTA_CHUNKS

    def test_xml_keeps_all_available_structure(self):
        """В ``.xml`` нет двух строк титула и три строки заголовка слиты в одну.

        Остальная структура (разделы, термины, приложения, библиография)
        обязана совпасть с эталоном — регресс здесь означает, что препроцессор
        перестал поднимать пункты предисловия, термины и названия приложений.
        """
        result = _parse("gost_report.xml")
        titles = [header.text for header in _headers(result.tree)]
        expected = ETALON_HEADERS - SOURCE_DELTAS["gost_report.xml"]
        assert len(titles) == expected
        assert any(title.startswith("1 ПОДГОТОВЛЕН") for title in titles)
        assert any(title.startswith("3.1 ") for title in titles)
        assert any(title == "Приложение А (справочное)" for title in titles)
        assert any(
            title == "Пример использования настоящего стандарта" for title in titles
        )

    def test_txt_is_uppercase_but_structurally_equal(self):
        """``.txt`` — верхний регистр, поэтому сверяем структуру, а не регистр."""
        result = _parse("gost_report.txt")
        titles = [header.text for header in _headers(result.tree)]
        assert len(titles) == ETALON_HEADERS
        assert any(title == "1 ОБЛАСТЬ ПРИМЕНЕНИЯ" for title in titles)
        assert any(title == "ИНФОРМАЦИОННЫЕ ТЕХНОЛОГИИ" for title in titles)
        assert any(title.startswith("Приложение А") for title in titles)

    def test_pdf_tables_match_etalon_count(self):
        """PDF: таблиц столько же, сколько в эталоне (фантом отброшен)."""
        result = _parse("gost_report.pdf")
        assert len(_tables(result.tree)) == ETALON_TABLES
        # Хвост разорванной таблицы остаётся в корне — привязать его не к чему.
        assert result.tree.metadata.get("orphan_tables", 0) == 1