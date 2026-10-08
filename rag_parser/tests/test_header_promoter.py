"""Тесты HeaderPromoter: скоринг + обе фикстуры (со стилями / без)."""

import pytest

from rag_parser.core.interfaces import DocumentNode
from rag_parser.parsers.dedoc_tree import build_full_tree
from rag_parser.parsers.header_promoter import promote_headers, score_node


def _node(text, annotations=None, node_type="paragraph"):
    return DocumentNode(
        type=node_type,
        level=0,
        text=text,
        children=[],
        metadata={"annotations_count": len(annotations or []),
                  "annotations": annotations or []},
    )


def _ann(name, start, end, value):
    return {"name": name, "start": start, "end": end, "value": value}


class TestScoring:
    def test_numbering_doc(self):
        score, level = score_node("Раздел 2. Показатели", [])
        assert score == 5.0  # +4 явный маркер раздела, +1 длина
        assert level == 1

    def test_long_razdel_promotes_despite_length(self):
        long_title = (
            "Раздел 5. Отчет совета директоров о результатах развития "
            "общества по приоритетным направлениям деятельности"
        )
        assert len(long_title) > 100
        score, level = score_node(long_title, [])
        assert score >= 4.0 and level == 1

    def test_numbering_paren_level(self):
        score, level = score_node("2) Функции потерь и метрики", [])
        assert score >= 3.0
        assert level == 2  # подчинённый уровень
        nospace, nospace_level = score_node("3)Подготовка датасета", [])
        assert nospace >= 3.0
        assert nospace_level == 2

    def test_numbering_trailing_dot_ignored(self):
        assert score_node("Раздел 2.", [])[1] == 1
        assert score_node("2.", [])[1] == 1
        assert score_node("2. x", [])[1] == 1
        assert score_node("2.3 Метрики", [])[1] == 2
        assert score_node("1.2.3 Детали", [])[1] == 3

    def test_paren_suppressed_with_razdel(self):
        score, level = score_node("3) пункт списка", [], allow_paren=False)
        assert score < 4.0 and level is None
        score2, level2 = score_node("Раздел 2. Показатели", [], allow_paren=False)
        assert score2 >= 4.0 and level2 == 1

    def test_numbering_dotted_level(self):
        score, level = score_node("2.3 Основные показатели", [])
        assert score >= 3.0
        assert level == 2
        _, level3 = score_node("1.2.3 Детали", [])
        assert level3 == 3

    def test_bold(self):
        text = "Короткий заголовок"
        base, _ = score_node(text, [])
        score, _ = score_node(
            text, [_ann("bold", 0, len(text), True)]
        )
        assert score == base + 2.0

    def test_bold_false_no_points(self):
        text = "Короткий заголовок"
        base, _ = score_node(text, [])
        score, _ = score_node(text, [_ann("bold", 0, len(text), "False")])
        assert score == base

    def test_size(self):
        text = "Заголовок"
        base, _ = score_node(text, [])
        score, _ = score_node(text, [_ann("size", 0, len(text), "16.0")])
        assert score == base + 1.5
        small, _ = score_node(text, [_ann("size", 0, len(text), "11.0")])
        assert small == base

    def test_caps(self):
        score, _ = score_node("ВАЖНОЕ ОБЪЯВЛЕНИЕ", [])
        base, _ = score_node("Важное объявление", [])
        assert score == base + 1.0

    def test_short(self):
        score, _ = score_node("Коротко", [])
        long_score, _ = score_node("x" * 100, [])
        assert score == long_score + 1.0

    def test_center(self):
        text = "Заголовок"
        base, _ = score_node(text, [])
        score, _ = score_node(text, [_ann("alignment", 0, len(text), "center")])
        assert score == base + 0.5

    def test_sentence_penalty(self):
        body = "Длинное предложение тела документа, которое заканчивается точкой. " * 3
        assert body.strip().endswith(".") and len(body) > 120
        score, _ = score_node(body, [])
        short = body[:50]
        short_score, _ = score_node(short, [])
        assert score < short_score
        assert score == short_score - 1.0 - 2.0 or score <= short_score - 2.0

    def test_empty(self):
        assert score_node("", []) == (0.0, None)


class TestPromote:
    def _root(self, children):
        return DocumentNode(type="document", level=0, text="",
                            children=children, metadata={})

    def test_threshold_override(self):
        children = [_node("Раздел 5. Итоги")]  # 3 + 1 = 4.0
        low = promote_headers(self._root(children), threshold=0.0)
        assert low.children[0].type == "header"
        high = promote_headers(self._root(children), threshold=8.0)
        assert high.children[0].type != "header"

    def test_default_threshold_from_config(self):
        children = [_node("Обычный абзац без признаков заголовка " * 5)]
        root = promote_headers(self._root(children))
        assert all(c.type != "header" for c in root.children)

    def test_multiline_split(self):
        body = "Текст тела. " * 30
        node = _node(f"Раздел 9. Заголовок\n{body}",
                     [_ann("bold", 0, 18, True),
                      _ann("size", 0, 18, "16"),
                      *[]])
        root = promote_headers(self._root([node]), threshold=4.0)
        types = [c.type for c in root.children]
        assert "header" in types
        header = next(c for c in root.children if c.type == "header")
        assert "Раздел 9" in header.text
        assert body[:20] not in header.text


def _parse_fixture(path, tmp_path, monkeypatch, threshold=4.0):
    from rag_parser.core.config import settings
    from rag_parser.parsers.dedoc_parser import DedocParser

    monkeypatch.setattr(
        settings, "DEDOC_OUTPUT_DIR", str(tmp_path / "output" / "dedoc_results")
    )
    return DedocParser(header_threshold=threshold).parse(path)


class TestFixtures:
    def test_styled_doc(self, header_docs, tmp_path, monkeypatch):
        result = _parse_fixture(header_docs["styled"], tmp_path, monkeypatch)
        assert result.tree is not None
        from rag_parser.parsers.dedoc_tree import collect_headers

        headers = collect_headers(result.tree)
        texts = [h.text for h in headers]
        assert any("Раздел 1" in t for t in texts)
        assert any("Метрики" in t for t in texts)

    def test_unstyled_doc(self, header_docs, tmp_path, monkeypatch):
        result = _parse_fixture(header_docs["unstyled"], tmp_path, monkeypatch)
        assert result.tree is not None
        from rag_parser.parsers.dedoc_tree import collect_headers

        headers = collect_headers(result.tree)
        texts = [h.text for h in headers]
        assert any("Раздел 2" in t for t in texts), texts
        assert any("Детализация" in t for t in texts), texts

    def test_threshold_changes_count(self, header_docs, tmp_path, monkeypatch):
        from rag_parser.parsers.dedoc_tree import collect_headers

        low = _parse_fixture(
            header_docs["unstyled"], tmp_path, monkeypatch, threshold=0.0
        )
        high = _parse_fixture(
            header_docs["unstyled"], tmp_path, monkeypatch, threshold=8.0
        )
        assert len(collect_headers(low.tree)) > len(collect_headers(high.tree))

    def test_annual_report_style(self, annual_report_doc, tmp_path, monkeypatch):
        """E2E в стиле 2023.docx: TOC подавлен, тело структурировано, lift."""
        from rag_parser.parsers.dedoc_tree import collect_headers

        result = _parse_fixture(annual_report_doc, tmp_path, monkeypatch)
        assert result.tree.metadata["toc_suppressed"] == 6
        assert result.tree.metadata["paren_suppressed"] == 3

        flat = []
        stack = [result.tree]
        while stack:
            current = stack.pop(0)
            flat.append(current)
            stack.extend(current.children)
        toc = [n for n in flat if "оглавления" in n.text]
        assert len(toc) == 6
        assert all(n.type != "header" for n in toc)

        headers = collect_headers(result.tree)
        by_text = {h.text: h for h in headers}
        r1 = next(h for t, h in by_text.items() if t.startswith("Раздел 1. Инфор"))
        r2 = next(h for t, h in by_text.items() if t.startswith("Раздел 2."))
        assert r1.level == 1 and r2.level == 1
        items = [n for n in flat if "утвержден" in n.text or "достоверность" in n.text]
        assert items and all(n.type != "header" for n in items)

        parents = _parent_map(result.tree)
        assert parents[id(r2)] is result.tree  # lift в корень
        sub = next(h for h in headers if h.text.startswith("2.1."))
        assert sub.level == 2
        assert parents[id(sub)] is r2


def _parent_map(root):
    mapping = {}

    def visit(node, parent):
        mapping[id(node)] = parent
        for child in node.children:
            visit(child, node)

    visit(root, None)
    return mapping


def _hnode(text):
    return DocumentNode(
        type="header", level=1, text=text, children=[], metadata={},
    )


class TestRebuild:
    def _root(self, children):
        return DocumentNode(type="document", level=0, text="",
                            children=children, metadata={})

    def test_buried_header_lifted_to_root(self):
        """Погребение из dedoc-вложенности: Раздел 2 ребёнком 3) → корень."""
        n3 = _node("3) годовой отчет утвержден собранием акционеров")
        n3.children.append(_node("Раздел 2. Финансовые показатели"))
        n1 = _node("Раздел 1. Информация о годовом отчете")
        n1.children.append(n3)
        root = promote_headers(self._root([n1]), threshold=4.0)
        parents = _parent_map(root)
        flat = []
        stack = [root]
        while stack:
            current = stack.pop(0)
            flat.append(current)
            stack.extend(current.children)
        razdel2 = next(n for n in flat if "Раздел 2" in n.text)
        assert razdel2.type == "header" and razdel2.level == 1
        assert parents[id(razdel2)] is root  # lift, а не погребение

    def test_level_zero_header_does_not_swallow(self):
        odd = DocumentNode(type="header", level=0, text="Странный",
                           children=[], metadata={})
        para = _node("Обычный абзац тела документа без признаков.")
        other = _hnode("Настоящий")
        root = promote_headers(self._root([odd, para, other]), threshold=8.0)
        parents = _parent_map(root)
        flat = []
        stack = [root]
        while stack:
            current = stack.pop(0)
            flat.append(current)
            stack.extend(current.children)
        found = next(n for n in flat if n.text.startswith("Обычный"))
        assert parents[id(found)] is root  # не поглощён L0-узлом

    def test_table_between_header_and_subheader(self):
        table = DocumentNode(type="table", level=0, text="| A |\n|---|\n| 1 |",
                             children=[], metadata={})
        h2 = _node("Раздел 2. Сведения")
        sub = _node("2.1. Реквизиты общества с ограниченной ответственностью")
        body = _node("Текст тела абзаца про реквизиты общества.")
        h2.children.extend([table, sub])
        sub.children.append(body)
        root = promote_headers(self._root([h2]), threshold=4.0)
        parents = _parent_map(root)
        flat = []
        stack = [root]
        while stack:
            current = stack.pop(0)
            flat.append(current)
            stack.extend(current.children)
        table_node = next(n for n in flat if n.type == "table")
        h2_node = next(n for n in flat if "Раздел 2" in n.text)
        assert parents[id(table_node)] is h2_node  # в Разделе 2, не в корне


class TestTocSuppression:
    def _root(self, children):
        return DocumentNode(type="document", level=0, text="",
                            children=children, metadata={})

    def _toc_fixture(self):
        children = [_node(f"Раздел {i}. Строка оглавления документа")
                    for i in range(1, 7)]
        children.append(_node("Раздел 1. Тело первого раздела документа"))
        children.append(_node("Текст тела первого раздела."))
        children.append(_node("Раздел 2. Тело второго раздела документа"))
        return children

    def test_toc_lines_stay_paragraph(self):
        root = promote_headers(self._root(self._toc_fixture()), threshold=4.0)
        assert root.metadata["toc_suppressed"] == 6
        toc = [c for c in root.children
               if "оглавления" in c.text or "Оглавление" in c.text
               or "Строка оглавления" in c.text]
        assert len(toc) == 6
        assert all(c.type != "header" for c in toc)

    def test_body_headers_promoted(self):
        root = promote_headers(self._root(self._toc_fixture()), threshold=4.0)
        headers = [c for c in root.children if c.type == "header"]
        texts = [h.text for h in headers]
        assert any("Тело первого" in t for t in texts)
        assert any("Тело второго" in t for t in texts)

    def test_short_series_not_suppressed(self):
        children = [_node(f"Раздел {i}. Короткая серия") for i in range(1, 4)]
        root = promote_headers(self._root(children), threshold=4.0)
        assert root.metadata["toc_suppressed"] == 0
        assert any(c.type == "header" for c in root.children)

    def test_paren_suppressed_with_razdel(self):
        children = [_node("Раздел 1. Первый"), _node("3) пункт перечня"),
                    _node("Раздел 2. Второй")]
        root = promote_headers(self._root(children), threshold=4.0)
        assert root.metadata["paren_suppressed"] >= 1
        flat = []
        stack = [root]
        while stack:
            current = stack.pop(0)
            flat.append(current)
            stack.extend(current.children)
        items = [n for n in flat if "пункт перечня" in n.text]
        assert items and all(n.type != "header" for n in items)

    def test_paren_suppressed_at_zero_threshold(self):
        """Супрессия абсолютна: порог 0.0 не промоутит N) в Раздел-документе."""
        children = [_node("Раздел 1. Первый"), _node("3) пункт перечня"),
                    _node("Раздел 2. Второй")]
        root = promote_headers(self._root(children), threshold=0.0)
        flat = []
        stack = [root]
        while stack:
            current = stack.pop(0)
            flat.append(current)
            stack.extend(current.children)
        items = [n for n in flat if "пункт перечня" in n.text]
        assert items and all(n.type != "header" for n in items)

    def test_paren_without_razdel_promotes(self):
        children = [_node("1) первый пункт"), _node("2) второй пункт")]
        root = promote_headers(self._root(children), threshold=4.0)
        assert root.metadata["paren_suppressed"] == 0
        assert sum(1 for c in root.children if c.type == "header") == 2
