"""Тесты breadcrumb: формат, лимит глубины, обогащение чанков."""

from rag_parser.chunkers.breadcrumb import (
    SEPARATOR,
    annotate_tree,
    breadcrumb_for_path,
    enrich_text,
)
from rag_parser.chunkers.hybrid import HybridRagChunker
from rag_parser.core.interfaces import DocumentNode


def _node(node_type, text, children=None, level=1):
    return DocumentNode(type=node_type, level=level, text=text,
                        children=children or [], metadata={})


def _tree():
    return DocumentNode(
        type="document", level=0, text="", children=[
            _node("header", "Раздел 2", [
                _node("header", "2.3 Основные показатели", [
                    _node("paragraph", "Текст про показатели."),
                    _node("table", "| A | B |\n|---|---|\n| 1 | 2 |"),
                ]),
            ]),
            _node("paragraph", "Текст без заголовков выше."),
        ], metadata={},
    )


class TestFormat:
    def test_separator(self):
        assert SEPARATOR == " > "
        assert breadcrumb_for_path(["A", "B"]) == "A > B"

    def test_last_three_levels(self):
        crumb = breadcrumb_for_path(["H1", "H2", "H3", "H4"])
        assert crumb == "H2 > H3 > H4"

    def test_short_path_whole(self):
        assert breadcrumb_for_path(["A", "B"]) == "A > B"

    def test_empty_none(self):
        assert breadcrumb_for_path([]) is None
        assert enrich_text("text", None) == "text"

    def test_enrich_format(self):
        assert enrich_text("текст", "Раздел 2 > 2.3 Основные показатели") == \
            "[Раздел 2 > 2.3 Основные показатели] текст"


class TestAnnotate:
    def test_paths(self):
        root = _tree()
        mapping = {id(node): crumb for node, crumb in annotate_tree(root)}
        h1 = root.children[0]
        h2 = h1.children[0]
        para = h2.children[0]
        lone = root.children[1]
        assert mapping[id(h1)] is None  # у заголовка верхнего уровня предков нет
        assert mapping[id(h2)] == "Раздел 2"
        assert mapping[id(para)] == "Раздел 2 > 2.3 Основные показатели"
        assert mapping[id(lone)] is None


class TestHybridEnrichment:
    def _chunker(self):
        return HybridRagChunker(text_strategy="token", chunk_size=100, overlap=10)

    def test_text_chunks_enriched(self):
        chunks = self._chunker().chunk("ignored", {"source": "d"}, tree=_tree())
        texts = [c for c in chunks if c.metadata["chunk_type"] == "text"]
        with_crumb = [c for c in texts if c.breadcrumb]
        assert with_crumb
        for chunk in with_crumb:
            assert chunk.text_for_embedding.startswith(f"[{chunk.breadcrumb}] ")
            assert chunk.text_for_embedding == f"[{chunk.breadcrumb}] {chunk.text}"
            assert chunk.metadata["breadcrumb"] == chunk.breadcrumb

    def test_table_chunks_both_enriched_no_pipes(self):
        chunks = self._chunker().chunk("ignored", {"source": "d"}, tree=_tree())
        tables = [c for c in chunks if c.metadata["chunk_type"] == "table"]
        assert tables
        for chunk in tables:
            assert chunk.breadcrumb == "Раздел 2 > 2.3 Основные показатели"
            assert chunk.text.startswith(f"[{chunk.breadcrumb}] ")
            assert chunk.text_for_embedding.startswith(f"[{chunk.breadcrumb}] ")
            assert "|" not in chunk.text_for_embedding

    def test_no_headers_no_enrichment(self):
        tree = DocumentNode(type="document", level=0, text="", children=[
            _node("paragraph", "Просто текст без заголовков.", level=0),
        ], metadata={})
        chunks = self._chunker().chunk("ignored", {"source": "d"}, tree=tree)
        assert chunks
        for chunk in chunks:
            if chunk.metadata["chunk_type"] == "text":
                assert chunk.breadcrumb is None
                assert chunk.text_for_embedding is None

    def test_markdown_fallback_without_tree(self):
        markdown = "# Заголовок\n\nТекст под заголовком."
        chunks = self._chunker().chunk(markdown, {"source": "d"})
        assert any(c.breadcrumb == "Заголовок" for c in chunks)
