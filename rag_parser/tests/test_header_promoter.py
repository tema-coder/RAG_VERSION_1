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
        # Orphan numbering на свежем трекере: уровень по числу точек,
        # ограниченный MAX_HEADER_LEVEL. Эталонный test_header_classification
        # из раздела 7.4 спеки ожидает для «1.1.1.1» -> 3 (обрезка),
        # а не 1. Прежнее поведение «orphan -> Level 1» противоречило спеке.
        assert second.calculate_level("1.1.1.1 Глубоко") == 3

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


class TestRazdelBaseHeaderReset:
    """РЕГРЕССИЯ 2023.docx: «Раздел N» — базовый заголовок Level 1 (сброс стека).

    ТЗ 1.2: «Раздел N» (где N — цифра) → базовый заголовок Level 1, сброс
    стека трекера. До фикса ``is_base_header`` не распознавал «Раздел N», и
    тот уходил в keyword-ветку: «Раздел 1» → L2 (ребро «Оглавление»),
    «Раздел 2» → L3, а 2.x получали orphan-уровень 2 и после пересборки
    оказывались под «Оглавление» — breadcrumb чанка 2.1 был «Оглавление:»
    вместо «Раздел 2. Сведения об Обществе».
    """

    @pytest.mark.parametrize(
        "text",
        [
            "Раздел 1. Информация о годовом отчете",
            "Раздел 2. Сведения об Обществе",
            "Раздел 14. Сведения о соблюдении Кодекса",
            "Раздел 1",          # без точки и текста — тоже базовый
            "Глава 2. Общая часть",
            "Часть 3. Дополнительные требования",
            "Том 1. Архив",
            "Книга 5. Приложения",
            "раздел 9. Основные факторы риска",  # регистронезависимо
        ],
    )
    def test_razdel_single_number_is_base_header(self, text):
        assert HeaderLevelTracker.is_base_header(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "Раздел 1.1",          # составной номер — вложенный триггер
            "Глава 1.1.1",
            "Раздел 1.1. Первое положение",
            "Разделение труда",    # не структурное слово
            "Раздельность",        # префикс ловиться не должен
            "1 Текст",             # цифра без КАПС/жирного — не базовый
            "Обычный текст",
        ],
    )
    def test_dotted_or_non_structural_is_not_base_header(self, text):
        assert HeaderLevelTracker.is_base_header(text) is False

    def test_razdel_resets_stack_to_level_1(self):
        """«Оглавление» → «Раздел 1» → «Раздел 2» → «2.1» = [1, 1, 1, 2].

        Воспроизводит порядок заголовков 2023.docx. До фикса давало
        [1, 2, 3, 2] (Раздел 1 → L2, Раздел 2 → L3, 2.1 → orphan L2).
        """
        tracker = HeaderLevelTracker()
        levels = [
            tracker.calculate_level("Оглавление:"),
            tracker.calculate_level("Раздел 1. Информация о годовом отчете"),
            tracker.calculate_level("Раздел 2. Сведения об Обществе"),
            tracker.calculate_level("2.1. Полное наименование: АО «СТАРТ»."),
            tracker.calculate_level("2.2. Сокращенное наименование: АО «СТАРТ»."),
            tracker.calculate_level("Раздел 3. Положение акционерного общества"),
            tracker.calculate_level("Раздел 9. Основные факторы риска"),
            tracker.calculate_level("9.1. Отраслевые риски"),
        ]
        assert levels == [1, 1, 1, 2, 2, 1, 1, 2]

    def test_razdel_does_not_deepen_after_subitems(self):
        """После 2.x новый «Раздел 3» снова Level 1, а не Level 3.

        Регрессия: до фикса «Раздел 3» шёл в keyword-ветку при глубоком
        стеке и получал cap=3, после пересборки становился ребёнком 2.9.
        """
        tracker = HeaderLevelTracker()
        sequence = [
            "Оглавление:",
            "Раздел 2. Сведения об Обществе",
            "2.1. Полное наименование",
            "2.9. Номер телефона",
            "Раздел 3. Положение в отрасли",
        ]
        levels = [tracker.calculate_level(text) for text in sequence]
        assert levels == [1, 1, 2, 2, 1]

    def test_dotted_razdel_still_nests(self):
        """«Раздел 1.1» — вложенный триггер (не базовый), поведение сохранено.

        Существующий инвариант: ``["1 Текст", "Раздел 1.1", "Глава 1.1.1"]``
        → [1, 2, 3]. Фикс «Раздел N» его не ломает: dotted-номер не base.
        """
        tracker = HeaderLevelTracker()
        levels = [
            tracker.calculate_level("1 Текст"),
            tracker.calculate_level("Раздел 1.1"),
            tracker.calculate_level("Глава 1.1.1"),
        ]
        assert levels == [1, 2, 3]


class TestRazdelEndToEnd:
    """Сквозная проверка promote_headers на структуре 2023.docx.

    Дерево строится вручную (без dedoc) из ключевых строк годового отчёта:
    титул, «Оглавление», 14 TOC-записей «Раздел N», тело «Раздел 1/2/9» с
    подпунктами 2.x и 9.x. Проверяется, что тело «Раздел N» — Level 1, а
    2.1 лежит под «Раздел 2» (breadcrumb «Раздел 2 …», НЕ «Оглавление:»).
    """

    @staticmethod
    def _build_2023_tree() -> DocumentNode:
        title = [
            DocumentNode(type="raw_text", level=0, text=t, children=[], metadata={})
            for t in ("ГОДОВОЙ ОТЧЕТ", "АКЦИОНЕРНОГО ОБЩЕСТВА «СТАРТ»", "ЗА 2023 ГОД")
        ]
        toc_entries = [
            DocumentNode(
                type="raw_text", level=0, text=f"Раздел {n}. Раздел №{n}",
                children=[], metadata={},
            )
            for n in range(1, 15)
        ]
        section1 = [
            DocumentNode(type="raw_text", level=0, text="Во исполнение закона…",
                         children=[], metadata={}),
        ]
        section2 = [
            DocumentNode(
                type="raw_text", level=0,
                text=f"2.{n}. Пункт 2.{n}",
                children=[], metadata={},
            )
            for n in range(1, 10)
        ]
        section9 = [
            DocumentNode(type="raw_text", level=0, text=f"9.{n}. Риск 9.{n}",
                         children=[], metadata={})
            for n in range(1, 4)
        ]
        children = title + [
            DocumentNode(type="raw_text", level=0, text="Оглавление:",
                         children=[], metadata={}),
            *toc_entries,
            DocumentNode(type="raw_text", level=0,
                         text="Раздел 1. Информация о годовом отчете",
                         children=[], metadata={}),
            *section1,
            DocumentNode(type="raw_text", level=0,
                         text="Раздел 2. Сведения об Обществе",
                         children=[], metadata={}),
            *section2,
            DocumentNode(type="raw_text", level=0,
                         text="Раздел 3. Положение акционерного общества в отрасли",
                         children=[], metadata={}),
            DocumentNode(type="raw_text", level=0,
                         text="Раздел 9. Основные факторы риска",
                         children=[], metadata={}),
            *section9,
        ]
        return DocumentNode(type="document", level=0, text="", children=children,
                            metadata={})

    def test_razdel_body_sections_are_level_1(self):
        from rag_parser.parsers.dedoc_tree import collect_headers

        root = promote_headers(self._build_2023_tree())
        headers = {h.text: h.level for h in collect_headers(root)}
        # Тело «Раздел N» — Level 1 (новый базовый), не L2/L3.
        assert headers["Раздел 1. Информация о годовом отчете"] == 1
        assert headers["Раздел 2. Сведения об Обществе"] == 1
        assert headers["Раздел 3. Положение акционерного общества в отрасли"] == 1
        assert headers["Раздел 9. Основные факторы риска"] == 1

    def test_subitems_nest_under_their_razdel(self):
        """2.1 — ребёнок «Раздел 2» (Level 2), а не «Оглавление»."""
        from rag_parser.parsers.dedoc_tree import collect_headers

        root = promote_headers(self._build_2023_tree())
        headers = collect_headers(root)
        by_text = {h.text: h for h in headers}
        assert by_text["2.1. Пункт 2.1"].level == 2
        assert by_text["9.1. Риск 9.1"].level == 2

    def test_toc_entries_are_suppressed(self):
        """14 TOC-записей «Раздел N» подавлены (series detection), не заголовки."""
        from rag_parser.parsers.dedoc_tree import collect_headers

        root = promote_headers(self._build_2023_tree())
        header_texts = [h.text for h in collect_headers(root)]
        # TOC-записи «Раздел N. Раздел №N» не должны стать заголовками.
        assert not any("Раздел №" in t for t in header_texts)
        assert root.metadata["toc_suppressed"] == 14

    def test_no_razdel_body_nests_under_oglavenie(self):
        """Ни один телесный «Раздел N» не ребёнок «Оглавление:».

        До фикса «Раздел 1» был ребёнком «Оглавление:» (L2), и breadcrumb
        его контента был «Оглавление: > Раздел 1 …». После фикса «Раздел 1»
        — top-level, его контент лежит прямо под ним.
        """
        from rag_parser.parsers.dedoc_tree import collect_headers

        root = promote_headers(self._build_2023_tree())
        oglavenie = next(
            h for h in collect_headers(root) if h.text == "Оглавление:"
        )
        oglavenie_children_text = " ".join(c.text for c in oglavenie.children)
        # Тело «Раздел 1» НЕ ребёнок «Оглавление:».
        assert "Раздел 1. Информация о годовом отчете" not in oglavenie_children_text
        assert "Раздел 2. Сведения об Обществе" not in oglavenie_children_text


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
        """Без аннотаций DOCX жирного нет — обычный текст заголовком не станет.

        В примере НЕ используются слова секций («Предисловие», «Введение»):
        они распознаются по правилу 3 независимо от жирности, и такой пример
        ничего не проверял бы.
        """
        assert classify_header("Обычный абзац текста", [])[0] is False

    def test_preface_is_section_word(self):
        """«Предисловие» — слово секции (правило 3), уровень 1.

        В txt-версии ГОСТ «Предисловие» идёт обычной строкой без всякой
        разметки; без этого слова оно оставалось абзацем, и все пункты
        предисловия уезжали в чанки с крошкой предыдущего заголовка.
        """
        is_header, level = classify_header("Предисловие", [])
        assert is_header is True and level == 1

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
    """Проблема 4: кап 3 уровней не должен просачиваться в разметочные форматы.

    Уровни заголовков разметочных форматов задаёт сам документ, но шкала
    приводится к эталонной: тело markdown/HTML обычно начинается на уровень
    глубже титульного блока (``#`` — название документа, ``##`` — первый
    раздел), а в эталонном дереве раздел «1 Область применения» — уровень 1
    (см. ``dedoc_tree.normalize_markup_levels``).
    """

    @staticmethod
    def _headers(tree):
        found = []
        stack = [tree]
        while stack:
            node = stack.pop(0)
            if node.type == "header":
                found.append(node)
            stack = list(node.children) + stack
        return found

    def test_md_levels_follow_section_numbering(self):
        """Уровень заголовка = глубина номера раздела (1 -> 1, 6.2.1 -> 3)."""
        import re
        from pathlib import Path

        from rag_parser.parsers.dedoc_parser import DedocParser

        source = Path(__file__).resolve().parent.parent.parent / "input" / "gost_report.md"
        if not source.exists():
            pytest.skip("файл отсутствует в input/")

        result = DedocParser().parse(source)
        levels = {}
        for header in self._headers(result.tree):
            match = re.match(r"^(\d+(?:\.\d+)*)\s", header.text)
            if match:
                levels.setdefault(match.group(1), header.level)

        assert levels["1"] == 1
        assert levels["5"] == 1
        assert levels["5.1"] == 2
        assert levels["6.2"] == 2
        assert levels["6.2.1"] == 3

    def test_md_levels_are_capped_at_three(self):
        """Кап 3 уровня: ``####`` в разметке не даёт уровень 4."""
        from pathlib import Path

        from rag_parser.parsers.dedoc_parser import DedocParser

        source = Path(__file__).resolve().parent.parent.parent / "input" / "gost_report.md"
        if not source.exists():
            pytest.skip("файл отсутствует в input/")

        result = DedocParser().parse(source)
        levels = {header.level for header in self._headers(result.tree)}
        assert max(levels) == 3
        assert {1, 2, 3} <= levels


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
