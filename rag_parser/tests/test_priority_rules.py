"""Тесты приоритетной системы распознавания заголовков (S → P0 → P5).

Спецификация — ``promt_for_right_work_with_head_page.txt``. Проверяют:

A. Построчные (параметризованные): классификация одной строки через
   ``classify_line`` + полный пайплайн (validate_chains → assign_ranks →
   build_tree) для chain-кандидатов.

B. Иерархия: ожидаемые деревья на последовательностях заголовков
   (ГОСТ, финансы, IEEE, «Введение → 1 Обзор»).

Старые тесты ``test_header_promoter.py`` не затрагивают — проверяют
только новую приоритетную систему.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from rag_parser.parsers.priority_rules import (
    CONFIDENCE_CHAIN,
    CONFIDENCE_EMPHASIS,
    CONFIDENCE_HARD,
    LineClassification,
    P0_APP,
    P0_NOTE,
    P0_WORD,
    P1_EN,
    P1_RU,
    assign_ranks,
    build_tree,
    classify_line,
    emphasis_from_annotations,
    emphasis_from_text_markers,
    is_caps,
    validate_chains,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _classify_full(text: str, *, emphasis=None, md_level=None) -> LineClassification:
    """Полный пайплайн на одной строке: classify → validate_chains → ranks.

    Для chain-кандидатов одиночная строка демоутится (chain < 2).
    """
    c = classify_line(text, emphasis=emphasis, md_level=md_level)
    cs = validate_chains([c])
    cs = assign_ranks(cs)
    cs, _ = build_tree(cs)
    return cs[0]


def _classify_sequence(lines, *, emphasis=None):
    """Полный пайплайн на последовательности: возвращает [(text, rule, rank, level)]."""
    cls = [classify_line(t, emphasis=emphasis) for t in lines]
    cls = validate_chains(cls)
    cls = assign_ranks(cls)
    cls, _ = build_tree(cls)
    return [(c.text, c.rule, c.rank, c.level) for c in cls if c.rule]


# --------------------------------------------------------------------------- #
# A. Построчные тесты — заголовки
# --------------------------------------------------------------------------- #

class TestHeadersPositive:
    """Строки, которые ДОЛЖНЫ быть заголовками."""

    @pytest.mark.parametrize("text,expected_rule", [
        ("Раздел 1. Общие положения", "P0_WORD"),
        ("Раздел 1.1. Первое положение", "P0_WORD"),
        ("Подраздел 2.3.1 Методы", "P0_WORD"),
        ("Глава 2. Общая часть", "P0_WORD"),
        ("Часть 3. Дополнительные требования", "P0_WORD"),
        ("1 ВВЕДЕНИЕ", "P2"),
        ("2.3 АНАЛИЗ ДАННЫХ", "P2"),
        ("I. ВНЕОБОРОТНЫЕ АКТИВЫ", "P2"),
        ("ОТЧЁТ О ФИНАНСОВОМ ПОЛОЖЕНИИ", "P4"),
        ("Введение", "P1"),
        ("ВВЕДЕНИЕ", "P1"),
        ("Introduction", "P1"),
        ("References:", "P1"),
        ("Заключение", "P1"),
        ("Содержание", "P1"),
        ("Приложение А (обязательное)", "P0_APP"),
        ("Приложение 1", "P0_APP"),
        ("Примечание 1. Основа подготовки", "P0_NOTE"),
    ])
    def test_is_header(self, text, expected_rule):
        c = _classify_full(text)
        assert c.rule == expected_rule, f"{text!r}: got {c.rule}, expected {expected_rule}"

    def test_p3_with_emphasis_true(self):
        """«1.1 Первое положение» с emphasis=True → P3 (confidence=emphasis)."""
        c = _classify_full("1.1 Первое положение", emphasis=True)
        assert c.rule == "P3"
        assert c.confidence == CONFIDENCE_EMPHASIS

    def test_p3_chain_candidate_accepted_in_sequence(self):
        """«1.1 Первое» в цепочке с «1 Введение» → P3 (confidence=chain)."""
        lines = ["1 Введение", "1.1 Первое", "1.2 Второе"]
        cls = [classify_line(t, emphasis=None) for t in lines]
        cls = validate_chains(cls)
        cls = assign_ranks(cls)
        cls, _ = build_tree(cls)
        rules = [c.rule for c in cls if c.rule]
        assert rules == ["P3", "P3", "P3"]
        assert all(c.confidence == CONFIDENCE_CHAIN for c in cls if c.rule)

    def test_p5_with_emphasis_true(self):
        """«I. Introduction» с emphasis=True → P5."""
        c = _classify_full("I. Introduction", emphasis=True)
        assert c.rule == "P5"
        assert c.confidence == CONFIDENCE_EMPHASIS

    def test_md_level_docling_header(self):
        """``#``-строка Docling (md_level=1) → rule=md, kind=md:1."""
        c = _classify_full("Заголовок из Docling", md_level=1)
        assert c.rule == "md"
        assert c.kind == "md:1"
        assert c.confidence == "source"


# --------------------------------------------------------------------------- #
# A. Построчные тесты — НЕ заголовки
# --------------------------------------------------------------------------- #

class TestHeadersNegative:
    """Строки, которые НЕ должны быть заголовками."""

    @pytest.mark.parametrize("text", [
        "Введение в методы анализа",
        "раздел 5 настоящего стандарта устанавливает требования",
        "12.05.2023 г. состоялось заседание",
        "3.5 мм",
        "A. Первое условие договора...",
        "Основные средства",
        "А. С. ПУШКИН",  # инициалы, не КАПС-заголовок
    ])
    def test_not_header(self, text):
        c = _classify_full(text)
        assert c.rule is None, f"{text!r}: got {c.rule}, expected None"

    @pytest.mark.parametrize("text", [
        "1.1.1 Настоящий стандарт устанавливает требования к изделиям.",
    ])
    def test_p3_single_demoted(self, text):
        """P3 emphasis=None одиночный → демоут (chain < 2)."""
        c = _classify_full(text)
        assert c.rule is None

    def test_p3_emphasis_false_rejected(self):
        """«1.1 Первое» с emphasis=False → не заголовок."""
        c = _classify_full("1.1 Первое положение", emphasis=False)
        assert c.rule is None

    def test_p5_emphasis_none_rejected(self):
        """«I. Introduction» с emphasis=None → не заголовок (P5 требует emphasis)."""
        c = _classify_full("I. Introduction", emphasis=None)
        assert c.rule is None


# --------------------------------------------------------------------------- #
# A. Подавления S
# --------------------------------------------------------------------------- #

class TestSuppression:
    """Правила S подавляют строку (suppressed=True, rule='S')."""

    @pytest.mark.parametrize("text", [
        "Примечание: здесь нюанс",
        "Примечание 1 — Текст примечания",
        "Примечание — текст",
        "1) Определить цель;",
        "Таблица 1 — Показатели",
        "ТАБЛИЦА 1 — ПОКАЗАТЕЛИ",
        "Рисунок 2 — Схема",
        "Table 1 — Indicators",
        "1 Область применения ........ 1",
    ])
    def test_suppressed(self, text):
        c = classify_line(text)
        assert c.suppressed is True
        assert c.rule == "S"

    def test_note_with_dot_is_header_not_suppressed(self):
        """«Примечание 1. Основа подготовки» — P0_NOTE, НЕ S_NOTE."""
        c = classify_line("Примечание 1. Основа подготовки")
        assert c.rule == "P0_NOTE"
        assert not c.suppressed


# --------------------------------------------------------------------------- #
# B. Иерархии
# --------------------------------------------------------------------------- #

class TestHierarchy:
    """Ожидаемые деревья на последовательностях заголовков."""

    def test_gost_sequence(self):
        """ГОСТ: 1 → 1.1 → 1.1.1 → 1.2 → 2 (уровни 1,2,3,2,1)."""
        lines = ["1 Введение", "1.1 Первое", "1.1.1 Глубже", "1.2 Второе", "2 Заключение"]
        result = _classify_sequence(lines)
        levels = [r[3] for r in result]
        assert levels == [1, 2, 3, 2, 1], f"levels: {levels}"

    def test_finance_sequence(self):
        """Финансы: ОТЧЁТ → I. → II. → ОТЧЁТ2 (уровни 1,2,2,1)."""
        lines = [
            "ОТЧЁТ О ФИНАНСОВОМ ПОЛОЖЕНИИ",
            "I. ВНЕОБОРОТНЫЕ АКТИВЫ",
            "II. ОБЯЗАТЕЛЬСТВА",
            "ОТЧЁТ О ПРИБЫЛЯХ И УБЫТКАХ",
        ]
        result = _classify_sequence(lines)
        levels = [r[3] for r in result]
        assert levels == [1, 2, 2, 1], f"levels: {levels}"

    def test_ieee_sequence_with_emphasis(self):
        """IEEE: I. → A. → B. → II. (с emphasis=True, уровни 1,2,2,1)."""
        lines = [
            "I. Introduction",
            "A. Subsection",
            "B. Another",
            "II. Methods",
        ]
        result = _classify_sequence(lines, emphasis=True)
        levels = [r[3] for r in result]
        assert levels == [1, 2, 2, 1], f"levels: {levels}"

    def test_vведение_then_1_obzor_are_siblings(self):
        """«Введение» (P1) и «1 Обзор» (P3) — соседи (оба уровень 1)."""
        lines = ["Введение", "1 Обзор", "1.1 Детали", "2 Заключение"]
        result = _classify_sequence(lines)
        # Введение L1, 1 Обзор L1, 1.1 L2, 2 L1
        levels = [r[3] for r in result]
        assert levels == [1, 1, 2, 1], f"levels: {levels}"

    def test_prilozhenie_closes_open_sections(self):
        """«Приложение А» (P0_APP, rank 1) закрывает открытые разделы."""
        lines = ["1 Введение", "1.1 Детали", "Приложение А (обязательное)", "Б. Дополнительно"]
        result = _classify_sequence(lines)
        # 1 L1, 1.1 L2, Приложение А L1 (закрыло 1.1), Б — не заголовок (нет emphasis)
        rules_levels = [(r[1], r[3]) for r in result]
        # Приложение А должно быть уровень 1
        app_idx = [i for i, r in enumerate(result) if "Приложение" in r[0]]
        assert app_idx, "Приложение должно быть заголовком"
        assert result[app_idx[0]][3] == 1, "Приложение должно быть уровень 1"

    def test_chast_glava_razdel_order(self):
        """«Часть 1 → Глава 1 → Раздел 1.1» — порядок появления, уровни 1,2,3."""
        lines = ["Часть 1. Общая", "Глава 1. Первая", "Раздел 1.1 Первый"]
        result = _classify_sequence(lines)
        levels = [r[3] for r in result]
        assert levels == [1, 2, 3], f"levels: {levels}"

    def test_max_depth_capped_at_3(self):
        """Глубина ограничена 3 уровнями: 1.1.1.1 → 3, не 4."""
        lines = ["1 Раздел", "1.1 Подраздел", "1.1.1 Пункт", "1.1.1.1 Подпункт"]
        result = _classify_sequence(lines)
        levels = [r[3] for r in result]
        assert levels == [1, 2, 3, 3], f"levels: {levels}"

    def test_header_without_text_before_subheader_stays_parent(self):
        """Заголовок без названия («Раздел 1.») остаётся заголовком-родителем.

        «1.1 Первый» после него — chain-кандидат; одиночный (нет серии)
        демоутится, но «Раздел 1.» остаётся заголовком L1. Это проверяет,
        что P0_WORD с ``allow_empty=True`` работает.
        """
        lines = ["Раздел 1.", "1.1 Первый"]
        result = _classify_sequence(lines)
        assert len(result) == 1  # «1.1 Первый» демоутнут (одиночный chain)
        assert result[0][0] == "Раздел 1."
        assert result[0][1] == "P0_WORD"
        assert result[0][3] == 1  # L1


# --------------------------------------------------------------------------- #
# emphasis-распознавание
# --------------------------------------------------------------------------- #

class TestEmphasis:
    """Распознавание emphasis из аннотаций и маркеров."""

    def test_emphasis_from_annotations_bold(self):
        """bold ≥90% → True."""
        anns = [{"name": "bold", "start": 0, "end": 20, "value": "True"}]
        assert emphasis_from_annotations(anns, "Жирный заголовок текст", None) is True

    def test_emphasis_from_annotations_size(self):
        """size ≥13pt и > base → True."""
        anns = [{"name": "size", "start": 0, "end": 10, "value": "16.0"}]
        assert emphasis_from_annotations(anns, "Крупный текст", 11.0) is True

    def test_emphasis_from_annotations_not_highlighted(self):
        """Аннотации есть, но не выделена → False."""
        anns = [{"name": "bold", "start": 0, "end": 3, "value": "True"}]  # 3/20 < 90%
        result = emphasis_from_annotations(anns, "Обычный абзац текст", 14.0)
        assert result is False

    def test_emphasis_from_annotations_none(self):
        """Нет аннотаций → None."""
        assert emphasis_from_annotations([], "текст", None) is None

    def test_emphasis_from_text_markers_bold(self):
        """``**Текст**`` → True."""
        assert emphasis_from_text_markers("**Жирный заголовок**") is True
        assert emphasis_from_text_markers("__BOLD__Заголовок__BOLD__") is True

    def test_emphasis_from_text_markers_none(self):
        """Без маркеров → None."""
        assert emphasis_from_text_markers("Обычный текст") is None


# --------------------------------------------------------------------------- #
# Регексы — синтаксис
# --------------------------------------------------------------------------- #

class TestRegexSyntax:
    """Проверка синтаксиса регексов (семантика — в тестах выше)."""

    def test_p0_word_matches_razdel(self):
        assert P0_WORD.match("Раздел 1. Общие положения")
        assert P0_WORD.match("РАЗДЕЛ 1. Общие положения")

    def test_p0_word_rejects_lowercase(self):
        """Регистр НЕ игнорировать: «раздел 1» — не матчит."""
        assert not P0_WORD.match("раздел 1. Общие положения")

    def test_p0_app_matches(self):
        assert P0_APP.match("Приложение А (обязательное)")
        assert P0_APP.match("Приложение 1")
        assert P0_APP.match("ПРИЛОЖЕНИЕ Б")

    def test_p0_note_requires_dot_after_number(self):
        """«Примечание N. Название» — точка после числа обязательна."""
        assert P0_NOTE.match("Примечание 1. Основа подготовки")
        assert not P0_NOTE.match("Примечание 1 — Основа подготовки")
        assert not P0_NOTE.match("Примечание: текст")

    def test_p1_ru_matches_section_words(self):
        assert P1_RU.match("Введение")
        assert P1_RU.match("ВВЕДЕНИЕ")
        assert P1_RU.match("Заключение:")
        assert P1_RU.match("Содержание.")
        assert not P1_RU.match("Введение в методы")

    def test_p1_en_matches_imrad(self):
        assert P1_EN.match("Introduction")
        assert P1_EN.match("Methods")
        assert P1_EN.match("Results and Discussion")
        assert P1_EN.match("References:")
        assert not P1_EN.match("Introduction to methods")

    def test_is_caps(self):
        assert is_caps("ВВЕДЕНИЕ") is True
        assert is_caps("ОТЧЁТ О ФИНАНСОВОМ ПОЛОЖЕНИИ") is True
        assert is_caps("Введение") is False
        assert is_caps("Обычный текст") is False
        assert is_caps("AB") is False  # <3 букв


# --------------------------------------------------------------------------- #
# Сквозная регрессия на gost_report.docx
# --------------------------------------------------------------------------- #

class TestGostReportRegression:
    """На gost_report.docx приоритетная система даёт ~34 заголовка (эталон 42).

    Расхождение с эталоном — намеренное: пункты предисловия («1 ПОДГОТОВЛЕН…»)
    и определения терминов («3.1 синтетический документ…») по спецификации
    НЕ заголовки (P3 emphasis=None, chain-кандидаты демоутятся). Эталон
    считал их заголовками по старому правилу 7 (нумерация).
    """

    def test_priority_gives_fewer_headers_than_old(self):
        """Приоритетная система даёт меньше заголовков, чем старая (42).

        Это ожидаемо: убраны ложные заголовки (пункты предисловия, термины).
        """
        from rag_parser.parsers.dedoc_parser import get_manager
        from rag_parser.parsers.dedoc_tree import build_full_tree, collect_headers
        from rag_parser.parsers.header_promoter import (
            promote_headers,
            promote_headers_priority,
            _presplit,
            _base_font_size,
        )

        mgr = get_manager()
        parsed = mgr.parse(
            "/home/z/my-project/upload/gost_report.docx",
            parameters={"with_attachments": "false", "structure_type": "tree"},
        )
        raw_tree = build_full_tree(parsed.to_api_schema().content.structure, None)

        # Базовый шрифт.
        split = []
        for ch in raw_tree.children:
            split.extend(_presplit(ch))
        scan = []

        def _c(ns):
            for i in ns:
                if i.type == "document":
                    _c(i.children)
                else:
                    scan.append(i)
                    _c(i.children)

        _c(split)
        base = _base_font_size([n for n in scan if not n.children])

        old = promote_headers(raw_tree, promote_non_headers=True)
        new = promote_headers_priority(raw_tree, source="docx", base_size=base)

        old_h = collect_headers(old)
        new_h = collect_headers(new)
        # Старая: 42. Новая: 34 (пункты предисловия и термины убраны).
        assert len(new_h) < len(old_h), (
            f"priority should give fewer headers than old "
            f"({len(new_h)} vs {len(old_h)})"
        )
        assert len(new_h) >= 30, (
            f"priority should keep at least 30 headers (корневые разделы + "
            f"подразделы + приложения), got {len(new_h)}"
        )
        # Все заголовки имеют metadata с rule/kind/rank/confidence.
        for h in new_h:
            assert "header_rule" in h.metadata
            assert "header_kind" in h.metadata
            assert "header_rank" in h.metadata
            assert "header_confidence" in h.metadata

    def test_priority_keeps_section_headers(self):
        """Корневые разделы ГОСТ (1-8, Приложение А) остаются заголовками L1."""
        from rag_parser.parsers.dedoc_parser import get_manager
        from rag_parser.parsers.dedoc_tree import build_full_tree, collect_headers
        from rag_parser.parsers.header_promoter import (
            promote_headers_priority,
            _presplit,
            _base_font_size,
        )

        mgr = get_manager()
        parsed = mgr.parse(
            "/home/z/my-project/upload/gost_report.docx",
            parameters={"with_attachments": "false", "structure_type": "tree"},
        )
        raw_tree = build_full_tree(parsed.to_api_schema().content.structure, None)
        split = []
        for ch in raw_tree.children:
            split.extend(_presplit(ch))
        scan = []

        def _c(ns):
            for i in ns:
                if i.type == "document":
                    _c(i.children)
                else:
                    scan.append(i)
                    _c(i.children)

        _c(split)
        base = _base_font_size([n for n in scan if not n.children])
        new = promote_headers_priority(raw_tree, source="docx", base_size=base)
        headers = {h.text.strip(): h for h in collect_headers(new)}

        # Корневые разделы — L1.
        for title in ["1 Область применения", "2 Нормативные ссылки",
                      "6 Структура документа", "8 Методы оценки"]:
            assert title in headers, f"«{title}» должен быть заголовком"
            assert headers[title].level == 1, f"«{title}» должен быть L1"

        # Подразделы — L2.
        assert "5.1 Первое положение" in headers
        assert headers["5.1 Первое положение"].level == 2
        assert "6.2 Элементы структуры" in headers
        assert headers["6.2 Элементы структуры"].level == 2

        # 6.2.1 — L3.
        assert "6.2.1 Категории данных" in headers
        assert headers["6.2.1 Категории данных"].level == 3
