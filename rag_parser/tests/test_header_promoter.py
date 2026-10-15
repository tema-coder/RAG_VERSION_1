"""Тесты детерминированного распознавания заголовков.

Скоринг и порог удалены: заголовком делает строгое совпадение с одним из
7 правил (см. docstring ``parsers/header_promoter.py``).
"""

import pytest

from rag_parser.core.interfaces import DocumentNode
from rag_parser.parsers.header_promoter import (
    MAX_HEADER_LEVEL,
    MIN_HEADER_FONT_SIZE,
    HeaderLevelTracker,
    classify_header,
    numbering_level,
    promote_headers,
    strip_bold_markers,
)


def _ann(name, value, length=40):
    return [{"name": name, "start": 0, "end": length, "value": value}]


class TestRule1FontSize:
    """Кегль >= 13pt И строго больше базового размера текста документа."""

    def test_size_above_base_is_header(self):
        is_header, level = classify_header(
            "Крупный заголовок", _ann("size", 16.0), base_size=11.0
        )
        assert is_header and level == 1

    def test_size_equal_to_base_is_not_header(self):
        """Строго больше: равный базовому кегль — обычный текст."""
        is_header, _ = classify_header(
            "Обычный текст документа", _ann("size", 11.0), base_size=11.0
        )
        assert is_header is False

    def test_size_below_threshold_ignored(self):
        """13pt — нижняя граница; 12pt даже при маленькой базе не заголовок."""
        is_header, _ = classify_header(
            "Текст", _ann("size", 12.0), base_size=9.0
        )
        assert is_header is False

    def test_threshold_constant_is_13(self):
        assert MIN_HEADER_FONT_SIZE == 13.0

    def test_without_base_size_font_rule_inactive(self):
        """Без базового размера правило не срабатывает (нет с чем сравнить)."""
        is_header, _ = classify_header("Текст", _ann("size", 20.0), base_size=None)
        assert is_header is False

    def test_comma_decimal_parsed(self):
        is_header, _ = classify_header(
            "Текст", _ann("size", "13,5"), base_size=10.0
        )
        assert is_header is True


class TestRule2StructuralWords:
    """Раздел / Глава / Часть / Том / Книга в начале строки."""

    @pytest.mark.parametrize(
        "text",
        [
            "Раздел 2. Показатели",
            "раздел 2. Показатели",
            "Глава 3 Основы",
            "Часть II Обзор",
            "Том 1",
            "Книга первая",
        ],
    )
    def test_structural_words_are_headers(self, text):
        is_header, level = classify_header(text)
        assert is_header is True
        assert level == 1

    def test_word_inside_line_is_not_enough(self):
        """«В Разделе 2 говорится...» — слово не в начале."""
        is_header, _ = classify_header("В Разделе 2 говорится о таблице")
        assert is_header is False

    def test_partial_word_not_matched(self):
        """«Раздельность» не должно ловиться как «Раздел»."""
        assert classify_header("Раздельность трудов")[0] is False


class TestRule3SectionWords:
    """Введение / Заключение / Приложение / Список литературы / Библиография."""

    @pytest.mark.parametrize(
        "text",
        ["Введение", "Заключение", "Приложение А", "Список литературы",
         "Библиография"],
    )
    def test_section_words_are_headers_level_one(self, text):
        is_header, level = classify_header(text)
        assert is_header is True
        assert level == 1

    def test_case_insensitive(self):
        assert classify_header("введение в работу")[0] is True


class TestRule4ContentWords:
    """Содержание / Оглавление / Table of Contents / Contents."""

    @pytest.mark.parametrize(
        "text", ["Содержание", "Оглавление", "Table of Contents", "Contents"]
    )
    def test_content_words_are_headers(self, text):
        is_header, level = classify_header(text)
        assert is_header is True and level == 1


class TestRule5Caps:
    """Текст целиком в верхнем регистре."""

    def test_caps_is_header(self):
        is_header, level = classify_header("МЕТОДОЛОГИЯ ИССЛЕДОВАНИЯ")
        assert is_header is True and level == 1

    def test_caps_with_digits_and_dots(self):
        assert classify_header("ГОСТ Р 99999-2026")[0] is True

    def test_title_case_is_not_header(self):
        assert classify_header("Методология исследования")[0] is False

    def test_single_letter_rejected(self):
        """Одна буква — слишком коротко, чтобы считать признаком."""
        assert classify_header("A")[0] is False

    def test_mixed_caps_not_header(self):
        assert classify_header("МЕТОДОЛОГИЯ исследования")[0] is False


class TestRule6CapsWithNumbering:
    """Нумерация + КАПС остатка: «2.3 АНАЛИЗ ДАННЫХ»."""

    def test_numbered_caps(self):
        is_header, level = classify_header("2.3 АНАЛИЗ ДАННЫХ")
        assert is_header is True
        assert level == 2

    def test_numbered_caps_deep(self):
        is_header, level = classify_header("6.2.3 ОБЛАСТИ ИСПОЛЬЗОВАНИЯ")
        assert is_header is True and level == 3


class TestRule7Numbering:
    """Числовая нумерация в начале.

    ``numbering_level`` — ЧИСТАЯ функция: она разбирает номер и ничего не
    знает про остальной документ, поэтому «1.1.1.1» даёт 4. Итоговый
    уровень заголовка считает ``HeaderLevelTracker`` по состоянию
    документа и ограничен тремя — см. ``TestDepthCap``.
    """

    @pytest.mark.parametrize(
        "text,raw_level",
        [
            ("1", 1),
            ("1)", 1),
            ("1. Название", 1),
            ("1.1", 2),
            ("1.1 Название", 2),
            ("2.3 Анализ", 2),
            ("1.1.1", 3),
            ("1.1.1 Название", 3),
            ("1.1.1.1", 4),
            ("1.1.1.1.1 Глубоко", 5),
        ],
    )
    def test_raw_level_from_dots(self, text, raw_level):
        assert numbering_level(text) == raw_level
        is_header, actual = classify_header(text)
        assert is_header is True and actual == raw_level

    def test_number_must_be_delimited(self):
        """«1abc» — цифра неотделима от текста, нумерации нет."""
        assert numbering_level("1abc") is None
        assert classify_header("1abc")[0] is False

    def test_no_numbering(self):
        assert numbering_level("Обычный текст") is None

    def test_paren_suppressed_when_disallowed(self):
        """Ветка N) подавляется в «Раздел N»-документах (allow_paren=False)."""
        assert numbering_level("1) пункт", allow_paren=False) is None
        assert classify_header("1) пункт", allow_paren=False)[0] is False
        # Остальная нумерация при этом работает
        assert numbering_level("1.1. Пункт", allow_paren=False) == 2


class TestPlainText:
    """Обычный текст не становится заголовком."""

    @pytest.mark.parametrize(
        "text",
        [
            "Абзац документа с обычным текстом.",
            "Это длинный текст без каких-либо признаков заголовка вообще.",
            "Таблица показывает значения 1, 2 и 3 по регионам страны.",
            "первый",
            "",
            "   ",
        ],
    )
    def test_not_a_header(self, text):
        is_header, level = classify_header(text)
        assert is_header is False
        assert level is None

    def test_bold_partially_is_not_enough(self):
        """Выделенный термин — не заголовок.

        Жирным должен быть почти весь текст строки (``bold``-аннотация
        покрывает >= ``BOLD_HEADING_COVERAGE``). У термина жирно только
        начало, поэтому строка остаётся абзацем.
        """
        text = "ПДн — персональные данные"
        annotations = [{"name": "bold", "start": 0, "end": 3, "value": "True"}]
        assert classify_header(text, annotations)[0] is False

    def test_bold_value_false_is_not_enough(self):
        text = "Жирный абзац"
        annotations = [
            {"name": "bold", "start": 0, "end": len(text), "value": "False"}
        ]
        assert classify_header(text, annotations)[0] is False

    def test_short_line_alone_is_not_enough(self):
        """Раньше короткая строка давала балл; теперь — нет."""
        assert classify_header("Коротко")[0] is False


class TestPromoteIntegration:
    """Интеграция: promote_headers применяет правила к дереву."""

    def _tree(self):
        from rag_parser.core.interfaces import DocumentNode

        def node(node_type, text, children=None, level=1):
            return DocumentNode(
                type=node_type, level=level, text=text,
                children=children or [], metadata={},
            )

        return node(
            "document", "",
            children=[
                node("raw_text", "1. Область применения"),
                node("raw_text", "Текст первого раздела."),
                node("raw_text", "1.1 Подраздел"),
                node("raw_text", "Текст подраздела."),
                node("raw_text", "Заключение"),
            ],
            level=0,
        )

    def test_headers_created_by_rules(self):
        result = promote_headers(self._tree())
        headers = {n.text.strip(): n.level for n in _walk_headers(result)}
        assert "1. Область применения" in headers
        assert headers["1. Область применения"] == 1
        assert "1.1 Подраздел" in headers
        assert headers["1.1 Подраздел"] == 2
        assert "Заключение" in headers

    def test_plain_text_stays_paragraph(self):
        result = promote_headers(self._tree())
        texts = [n.text for n in _walk_all(result)]
        assert any("Текст первого раздела." in t for t in texts)
        header_texts = {n.text.strip() for n in _walk_headers(result)}
        assert not any("Текст первого раздела." in h for h in header_texts)

    def test_threshold_argument_is_ignored(self):
        """DEPRECATED-параметр принимается, но на результат не влияет."""
        low = promote_headers(self._tree(), threshold=0.0)
        high = promote_headers(self._tree(), threshold=8.0)
        assert [n.text for n in _walk_all(low)] == [n.text for n in _walk_all(high)]
        assert [n.level for n in _walk_all(low)] == [n.level for n in _walk_all(high)]


def _walk_all(node):
    yield node
    for child in node.children:
        yield from _walk_all(child)


def _walk_headers(node):
    for item in _walk_all(node):
        if item.type == "header":
            yield item

class TestHeaderLevelTracker:
    """Уровень по состоянию документа: углубление против соседства."""

    @pytest.mark.parametrize(
        "sequence,expected_levels",
        [
            # Базовые случаи
            (["1 Текст", "1.1 Текст", "1.2 Текст"], [1, 2, 2]),  # Сброс
            (["1 Текст", "1.1 Текст", "1.1.1 Текст"], [1, 2, 3]),  # Углубление
            (
                ["1 Текст", "1.1 Текст", "1.1.1 Текст", "1.1.1.1 Текст"],
                [1, 2, 3, 3],
            ),  # Обрезка
            # Сложные случаи
            (
                ["1 Текст", "1.1 Текст", "1.2 Текст", "1.2.1 Текст", "1.2.2 Текст"],
                [1, 2, 2, 3, 3],
            ),
            (
                ["1 Текст", "1.1 Текст", "2 Текст", "2.1 Текст"],
                [1, 2, 1, 2],
            ),  # Новый базовый
            # Триггеры без нумерации
            (["1 Текст", "**Подраздел**", "***Вложенный***"], [1, 2, 3]),
            (["1 Текст", "Раздел 1.1", "Глава 1.1.1"], [1, 2, 3]),
        ],
    )
    def test_header_level_calculation(self, sequence, expected_levels):
        tracker = HeaderLevelTracker()
        levels = [tracker.calculate_level(text) for text in sequence]
        assert levels == expected_levels

    def test_max_level_constant(self):
        assert MAX_HEADER_LEVEL == 3

    def test_siblings_do_not_deepen(self):
        """«1.1» и «1.2» — соседи одного уровня, а не цепочка."""
        tracker = HeaderLevelTracker()
        assert tracker.calculate_level("1 Введение") == 1
        assert tracker.calculate_level("1.1 Первый") == 2
        assert tracker.calculate_level("1.2 Второй") == 2
        assert tracker.calculate_level("1.3 Третий") == 2

    def test_depth_is_capped(self):
        """Ни один заголовок не опускается глубже третьего уровня."""
        tracker = HeaderLevelTracker()
        levels = [
            tracker.calculate_level(text)
            for text in ("1", "1.1", "1.1.1", "1.1.1.1", "1.1.1.1.1", "1.1.1.1.1.1")
        ]
        assert levels == [1, 2, 3, 3, 3, 3]

    def test_siblings_inherit_the_previous_level(self):
        """РЕГРЕССИЯ: сосед наследует УРОВЕНЬ, а не сбрасывается в 2.

        После скользящего окна стек равен [(2, «6.2»), (3, «6.2.2»)].
        Раньше соседство с «6.2.2» жёстко давало 2, и «6.2.3» получал
        уровень 2 вместо 3 — уровень «прыгал» внутри одного оглавления.
        """
        tracker = HeaderLevelTracker()
        levels = [
            tracker.calculate_level(text)
            for text in (
                "6 Структура документа",
                "6.2 Элементы структуры",
                "6.2.1 Категории данных",
                "6.2.2 Квалификаторы идентификации",
                "6.2.3 Области использования",
            )
        ]
        assert levels == [1, 2, 3, 3, 3]

    def test_tracker_is_per_document(self):
        """Новый экземпляр начинает с чистого состояния."""
        first = HeaderLevelTracker()
        for text in ("1", "1.1", "1.1.1"):
            first.calculate_level(text)
        second = HeaderLevelTracker()
        assert second.level_stack == []
        assert second.calculate_level("1.1.1.1 Глубоко") == 1

    def test_reset_clears_stack(self):
        tracker = HeaderLevelTracker()
        tracker.calculate_level("1 Введение")
        tracker.calculate_level("1.1 Первый")
        tracker.reset()
        assert tracker.level_stack == []

    def test_base_header_detection(self):
        assert HeaderLevelTracker.is_base_header("1 ОБЛАСТЬ ПРИМЕНЕНИЯ") is True
        assert HeaderLevelTracker.is_base_header("1 **Область применения**") is True
        assert HeaderLevelTracker.is_base_header("1 Текст") is False

    def test_trigger_detection(self):
        assert HeaderLevelTracker.detect_trigger("1.1 Текст") == "numbered"
        assert HeaderLevelTracker.detect_trigger("**Подраздел**") == "bold"
        assert HeaderLevelTracker.detect_trigger("Раздел 1.1") == "keyword"
        assert HeaderLevelTracker.detect_trigger("Обычный текст") is None

    def test_extract_number(self):
        assert HeaderLevelTracker.extract_number("1.2.1 Название") == "1.2.1"
        assert HeaderLevelTracker.extract_number("Раздел 1.1") == ""


class TestBoldTrigger:
    """Триггер жирного текста и его очистка из результата."""

    @pytest.mark.parametrize(
        "text,cleaned",
        [
            ("**Подраздел**", "Подраздел"),
            ("__BOLD__Текст__BOLD__ и конец", "Текст и конец"),
            ("<b>Жирно</b>", "Жирно"),
            ("обычный текст", "обычный текст"),
            # Инлайновый жирный внутри строки маркерами НЕ считается:
            # снимать края можно только если строка обёрнута целиком.
            ("**жирный** внутри", "**жирный** внутри"),
        ],
    )
    def test_strip_bold_markers(self, text, cleaned):
        assert strip_bold_markers(text) == cleaned

    def test_bold_line_is_a_header(self):
        is_header, level = classify_header("**Подраздел**")
        assert is_header is True
        # Уровень задаёт трекер, а не классификатор
        assert level is None


class TestDepthCap:
    """Итоговый уровень в дереве ограничен тремя."""

    def _document(self, lines):
        return DocumentNode(
            type="document",
            level=0,
            text="",
            children=[
                DocumentNode(
                    type="raw_text", level=1, text=text, children=[], metadata={}
                )
                for text in lines
            ],
            metadata={},
        )

    def _levels(self, root):
        found = []

        def walk(node):
            if node.type == "header":
                found.append((node.text.strip(), node.level))
            for child in node.children:
                walk(child)

        walk(root)
        return found

    @pytest.mark.parametrize(
        "lines,expected",
        [
            (["1 Текст", "1.1 Текст", "1.2 Текст"], [1, 2, 2]),
            (["1 Текст", "1.1 Текст", "1.1.1 Текст", "1.1.1.1 Текст"], [1, 2, 3, 3]),
            (["1 Текст", "1.1 Текст", "2 Текст", "2.1 Текст"], [1, 2, 1, 2]),
            (["1 Текст", "**Подраздел**", "***Вложенный***"], [1, 2, 3]),
        ],
    )
    def test_document_levels(self, lines, expected):
        result = promote_headers(self._document(lines))
        assert [level for _, level in self._levels(result)] == expected

    def test_deep_numbering_is_capped_in_tree(self):
        """"1.1.1.1" — четвёртый уровень по точкам, но в дереве максимум 3."""
        result = promote_headers(
            self._document(["1", "1.1", "1.1.1", "1.1.1.1 Глубоко"])
        )
        levels = [level for _, level in self._levels(result)]
        assert levels == [1, 2, 3, 3]
        assert max(levels) == MAX_HEADER_LEVEL


class TestBoldFromDocxAnnotations:
    """Проблема 1: жирность в DOCX приходит аннотациями, не текстом."""

    def _bold(self, length, value="True"):
        return [{"name": "bold", "start": 0, "end": length, "value": value}]

    def test_fully_bold_line_is_header(self):
        text = "Предисловие"
        assert classify_header(text, self._bold(len(text)))[0] is True

    def test_numbered_bold_is_header_level_one(self):
        text = "1 Область применения"
        is_header, level = classify_header(text, self._bold(len(text)))
        assert is_header is True and level == 1

    def test_bold_cover_threshold(self):
        """Граница 90%: 8 из 10 символов жирно — ещё термин, 10 из 10 — заголовок.

        Текст без цифр в начале, иначе сработало бы правило 7 (нумерация).
        """
        text = "абвгдеёжзий"
        almost = [{"name": "bold", "start": 0, "end": 8, "value": "True"}]
        whole = [{"name": "bold", "start": 0, "end": 10, "value": "True"}]
        assert classify_header(text, almost)[0] is False
        assert classify_header(text, whole)[0] is True

    def test_no_annotations_means_no_bold(self):
        assert classify_header("Предисловие", [])[0] is False

    def test_bold_header_in_docx(self, tmp_path):
        """Сквозной тест на настоящем DOCX без стилей Heading."""
        docx = pytest.importorskip("docx")
        from rag_parser.parsers.dedoc_parser import DedocParser

        document = docx.Document()
        run = document.add_paragraph().add_run("Предисловие")
        run.bold = True
        document.add_paragraph().add_run("Обычный абзац текста без жирности.")
        term = document.add_paragraph()
        term_run = term.add_run("ПДн")
        term_run.bold = True
        term.add_run(" — персональные данные.")
        source = tmp_path / "bold.docx"
        document.save(source)

        tree = DedocParser().parse(source).tree

        def collect(node):
            found = []
            if node.type == "header":
                found.append(node.text.strip())
            for child in node.children:
                found.extend(collect(child))
            return found

        titles = collect(tree)
        assert "Предисловие" in titles
        # выделенный термин заголовком не стал
        assert not any("ПДн" in title for title in titles)


class TestTocSuppression:
    """Проблема 2: записи оглавления не должны становиться заголовками."""

    def _document(self, lines):
        return DocumentNode(
            type="document",
            level=0,
            text="",
            children=[
                DocumentNode(
                    type="raw_text", level=1, text=text, children=[], metadata={}
                )
                for text in lines
            ],
            metadata={},
        )

    def _titles(self, root):
        found = []

        def walk(node):
            if node.type == "header":
                found.append(node.text.strip())
            for child in node.children:
                walk(child)

        walk(root)
        return found

    def test_toc_entries_are_paragraphs(self):
        lines = ["Содержание", "1 Область применения … 1", "5 Общие положения … 3",
                 "5.1 Первое положение … 4", "5.2 Второе положение … 5",
                 "6 Структура документа … 5", "7 Правила применения … 9",
                 "1 Область применения", "Текст первого раздела."]
        result = promote_headers(self._document(lines))
        titles = self._titles(result)
        assert not any("…" in title for title in titles), titles
        # настоящие разделы остались заголовками
        assert "1 Область применения" in titles

    def test_toc_heading_itself_stays_header(self):
        result = promote_headers(self._document(["Содержание", "1 Раздел … 1"]))
        assert "Содержание" in self._titles(result)

    def test_short_list_is_not_treated_as_toc(self):
        """Меньше TOC_SERIES_MIN записей — это обычный текст, не оглавление."""
        lines = ["Содержание", "1 Раздел … 1", "1 Тема", "Текст."]
        result = promote_headers(self._document(lines))
        assert "Содержание" in self._titles(result)

    def test_numbered_text_without_page_marker_is_kept(self):
        """Номер без маркера страницы — обычный заголовок, а не запись ТОС."""
        lines = ["Содержание", "1 Раздел … 1", "2 Раздел … 2", "3 Раздел … 3",
                 "4 Раздел … 4", "5 Раздел … 5", "1 Область применения"]
        result = promote_headers(self._document(lines))
        assert "1 Область применения" in self._titles(result)

    def test_counter_is_reported(self):
        lines = ["Содержание"] + [f"{n} Раздел … {n}" for n in range(1, 7)]
        result = promote_headers(self._document(lines))
        assert result.metadata.get("toc_suppressed") == 6

    def test_real_txt_has_no_toc_headers(self):
        """Проверка на настоящем gost_report.txt."""
        from pathlib import Path

        from rag_parser.parsers.dedoc_parser import DedocParser

        source = Path(__file__).resolve().parent.parent.parent / "input" / "gost_report.txt"
        if not source.exists():
            pytest.skip("файл отсутствует в input/")

        titles = self._titles(DedocParser().parse(source).tree)
        assert not any("…" in title for title in titles)
        assert len(titles) < 56  # раньше 19 записей оглавления были заголовками


class TestMarkupFormatsKeepAuthoritativeLevels:
    """Проблема 4: кап 3 уровней не должен просачиваться в разметочные форматы."""

    def test_md_levels_match_headings(self):
        """Уровни .md совпадают с глубиной ``#`` в исходнике."""
        import re
        from pathlib import Path

        from rag_parser.parsers.dedoc_parser import DedocParser

        source = Path(__file__).resolve().parent.parent.parent / "input" / "gost_report.md"
        if not source.exists():
            pytest.skip("файл отсутствует в input/")

        depths = [
            len(mark.group(1))
            for line in source.read_text(encoding="utf-8").split("\n")
            if (mark := re.match(r"^(#+) ", line))
        ]

        result = DedocParser().parse(source)
        levels = []
        stack = [result.tree]
        while stack:
            node = stack.pop(0)
            if node.type == "header":
                levels.append(node.level)
            stack = list(node.children) + stack

        assert len(levels) == len(depths)
        assert sorted(levels) == sorted(depths)
        assert max(levels) >= 4, "уровень 4 из h4 обязан сохраниться"


class TestTxtNesting:
    """Проблема 5: вложенность в TXT."""

    def test_real_txt_nested_levels(self):
        from pathlib import Path

        from rag_parser.parsers.dedoc_parser import DedocParser

        source = Path(__file__).resolve().parent.parent.parent / "input" / "gost_report.txt"
        if not source.exists():
            pytest.skip("файл отсутствует в input/")

        result = DedocParser().parse(source)
        levels = set()
        stack = [result.tree]
        while stack:
            node = stack.pop(0)
            if node.type == "header":
                levels.add(node.level)
            stack = list(node.children) + stack
        assert {1, 2, 3} <= levels
