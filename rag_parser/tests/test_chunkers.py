import re

import pytest

from rag_parser.chunkers.base import count_tokens, get_encoding
from rag_parser.chunkers.hybrid import HybridRagChunker
from rag_parser.chunkers.recursive import DEFAULT_SEPARATORS, RecursiveChunker
from rag_parser.chunkers.semantic import SemanticChunkerRAG
from rag_parser.chunkers.sentence import SentenceChunkerRAG
from rag_parser.chunkers.table import (
    TableChunkerRAG,
    _clean_table_header,
    _is_markdown_table,
    table_to_natural_language,
)
from rag_parser.chunkers.token import TokenChunkerRAG
from rag_parser.core.interfaces import Chunk

SENTENCE_END_RE = re.compile(r"[.!?]+\s+")


def _sentences(text: str):
    return [part for part in SENTENCE_END_RE.split(text.strip()) if part]


def assert_chunk_invariants(chunks, source="document"):
    assert chunks
    for index, chunk in enumerate(chunks):
        assert isinstance(chunk, Chunk)
        assert chunk.index == index
        assert chunk.text.strip()
        assert chunk.source == source
        assert chunk.tokens > 0
        assert chunk.metadata["chunk_index"] == index
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


class TestSharedHelpers:
    def test_count_tokens(self):
        assert count_tokens("") == 0
        assert count_tokens("hello world") > 0
        assert count_tokens("hello world") == len(
            get_encoding().encode("hello world", disallowed_special=())
        )

    def test_special_tokens_do_not_crash(self):
        assert count_tokens("<|endoftext|> and <|im_start|>") > 0


class TestTokenChunkerRAG:
    def test_chunk_sizes_match_budget(self, sample_text):
        chunks = TokenChunkerRAG(chunk_size=50, overlap=0).chunk(sample_text, {})
        assert_chunk_invariants(chunks)
        assert len(chunks) > 1
        assert all(chunk.tokens <= 50 for chunk in chunks[:-1])
        assert chunks[0].tokens == 50
        assert chunks[0].metadata["chunker"] == "token"

    def test_overlap_is_applied(self, sample_text):
        without = TokenChunkerRAG(chunk_size=60, overlap=0).chunk(sample_text, {})
        with_overlap = TokenChunkerRAG(chunk_size=60, overlap=20).chunk(
            sample_text, {}
        )
        assert len(with_overlap) > len(without)

    def test_small_text_single_chunk(self):
        chunks = TokenChunkerRAG(chunk_size=500).chunk("Короткий текст.", {})
        assert len(chunks) == 1
        assert chunks[0].text == "Короткий текст."

    def test_empty_text(self):
        assert TokenChunkerRAG().chunk("", {}) == []
        assert TokenChunkerRAG().chunk("   \n ", {}) == []

    def test_metadata_contains_source(self, sample_text):
        chunks = TokenChunkerRAG(chunk_size=100).chunk(
            sample_text, {"source": "report.pdf", "format": "pdf"}
        )
        assert chunks[0].source == "report.pdf"
        assert chunks[0].metadata["format"] == "pdf"
        assert "source" not in chunks[0].metadata

    @pytest.mark.parametrize("chunk_size,overlap", [(0, 0), (100, 100), (100, 150)])
    def test_invalid_parameters(self, chunk_size, overlap):
        with pytest.raises(ValueError):
            TokenChunkerRAG(chunk_size=chunk_size, overlap=overlap)


class TestSentenceChunkerRAG:
    def test_no_broken_sentences(self, sample_text):
        chunks = SentenceChunkerRAG(chunk_size=60).chunk(sample_text, {})
        assert_chunk_invariants(chunks)
        for chunk in chunks:
            for sentence in _sentences(chunk.text):
                assert sentence in sample_text

    def test_sentence_metadata(self, sample_text):
        chunks = SentenceChunkerRAG(chunk_size=60).chunk(sample_text, {})
        for chunk in chunks:
            assert chunk.metadata["chunker"] == "sentence"
            assert chunk.tokens <= 60

    def test_short_text(self):
        chunks = SentenceChunkerRAG(chunk_size=500).chunk("Одно предложение.", {})
        assert len(chunks) == 1

    def test_empty_text(self):
        assert SentenceChunkerRAG().chunk("", {}) == []
        assert SentenceChunkerRAG().chunk("\n\t", {}) == []

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError):
            SentenceChunkerRAG(chunk_size=0)


RECURSIVE_DOC = (
    "Машинное обучение изучает алгоритмы, которые учатся на данных. "
    "Модели достигают высокой точности на проверочных наборах.\n"
    "Классические алгоритмы остаются интерпретируемыми и быстрыми.\n"
    "Глубокое обучение требует больших объёмов размеченных данных.\n"
    "\n"
    "Современные системы достигают уровня человека в ряде задач. "
    "Это утверждение проверяется на практике.\n"
    "\n"
    "Отдельный раздел про инфраструктуру и эксплуатацию моделей. "
    "В нём описаны только практические детали."
) * 4


class TestRecursiveChunker:
    def test_defaults(self):
        chunker = RecursiveChunker()
        assert chunker.chunk_size == 200
        assert chunker.overlap == 20
        assert chunker.separators == DEFAULT_SEPARATORS
        assert chunker.name == "recursive"

    def test_paragraph_boundaries_are_kept(self):
        chunks = RecursiveChunker(chunk_size=120, overlap=0).chunk(RECURSIVE_DOC, {})
        assert_chunk_invariants(chunks)
        assert len(chunks) > 1
        # A paragraph that fits the budget is never split by the line level.
        first_paragraph = (
            "Машинное обучение изучает алгоритмы, которые учатся на данных. "
            "Модели достигают высокой точности на проверочных наборах."
        )
        assert any(first_paragraph in chunk.text for chunk in chunks)

    def test_sentences_are_kept_when_paragraph_does_not_fit(self):
        small = RecursiveChunker(chunk_size=40, overlap=0).chunk(RECURSIVE_DOC, {})
        assert_chunk_invariants(small)
        for chunk in small:
            for sentence in _sentences(chunk.text):
                assert sentence.strip() in RECURSIVE_DOC

    def test_overlap_is_applied(self):
        without = RecursiveChunker(chunk_size=120, overlap=0).chunk(RECURSIVE_DOC, {})
        with_overlap = RecursiveChunker(chunk_size=120, overlap=20).chunk(
            RECURSIVE_DOC, {}
        )
        assert len(with_overlap) >= len(without)
        # Consecutive chunks share the tail of the previous chunk.
        assert with_overlap[1].text[:20] in with_overlap[0].text

    def test_overlap_never_exceeds_previous_chunk(self):
        chunks = RecursiveChunker(chunk_size=60, overlap=20).chunk(RECURSIVE_DOC, {})
        encoding = get_encoding()
        for previous, current in zip(chunks, chunks[1:]):
            tail = encoding.decode(
                encoding.encode(previous.text, disallowed_special=())[-20:]
            )
            assert current.text.startswith(tail)

    def test_metadata(self, sample_text):
        chunks = RecursiveChunker(chunk_size=100, overlap=10).chunk(
            sample_text, {"source": "report.pdf", "format": "pdf"}
        )
        chunk = chunks[0]
        assert chunk.metadata["chunker"] == "recursive"
        assert chunk.metadata["chunk_size"] == 100
        assert chunk.metadata["overlap"] == 10
        assert chunk.metadata["separators"] == DEFAULT_SEPARATORS
        assert chunk.source == "report.pdf"
        assert chunk.metadata["format"] == "pdf"
        assert "source" not in chunk.metadata

    def test_token_count_matches_text(self):
        chunks = RecursiveChunker(chunk_size=80, overlap=5).chunk(RECURSIVE_DOC, {})
        assert all(chunk.tokens == count_tokens(chunk.text) for chunk in chunks)

    def test_custom_separators(self):
        chunks = RecursiveChunker(
            chunk_size=60, overlap=0, separators=["\n\n", " "]
        ).chunk(RECURSIVE_DOC, {})
        assert_chunk_invariants(chunks)

    def test_short_text(self):
        chunks = RecursiveChunker().chunk("Одно предложение.", {})
        assert len(chunks) == 1

    def test_empty_text(self):
        assert RecursiveChunker().chunk("", {}) == []
        assert RecursiveChunker().chunk("  \n ", {}) == []

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"chunk_size": 0},
            {"chunk_size": 100, "overlap": 100},
            {"chunk_size": 100, "overlap": 150},
            {"chunk_size": 100, "separators": []},
        ],
    )
    def test_invalid_parameters(self, kwargs):
        with pytest.raises(ValueError):
            RecursiveChunker(**kwargs)


def _hub_reachable() -> bool:
    import socket

    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
        return True
    except OSError:
        return False


class TestSemanticChunkerRAG:
    @pytest.mark.slow
    def test_smoke_small_text(self):
        if not _hub_reachable():
            pytest.skip("no network: embedding model cannot be downloaded")
        try:
            chunks = SemanticChunkerRAG(chunk_size=200, threshold=0.5).chunk(
                "First sentence about cats. Second sentence about cats. "
                "Unrelated text about quantum physics formulas. "
                "More formulas and equations here. ",
                {},
            )
        except Exception as exc:  # noqa: BLE001 - model download may fail
            pytest.skip(f"embedding model unavailable: {exc}")
        assert_chunk_invariants(chunks)
        # Semantic chunks are sentence groups, not character fragments.
        assert all(len(chunk.text) > 10 for chunk in chunks)
        assert chunks[0].metadata["chunker"] == "semantic"
        assert chunks[0].metadata["threshold"] == 0.5

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            SemanticChunkerRAG(threshold=0.0)
        with pytest.raises(ValueError):
            SemanticChunkerRAG(threshold=1.0)

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError):
            SemanticChunkerRAG(chunk_size=0)

    def test_empty_text_no_model_needed(self):
        assert SemanticChunkerRAG().chunk("", {}) == []
        assert SemanticChunkerRAG().chunk("  ", {}) == []


def _table_7() -> str:
    return "| A | B |\n|---|---|\n" + "\n".join(
        f"| {index} | {index * 2} |" for index in range(7)
    )


class TestTableDetection:
    def test_detects_markdown_table(self):
        assert _is_markdown_table(_table_7())
        assert not _is_markdown_table("Just some text.\nMore text.")
        assert not _is_markdown_table("| only | one row |")

    def test_cleans_placeholder_header(self):
        dirty = "|  |  |\n|---|---|\n| H1 | H2 |\n| a | b |"
        assert _clean_table_header(dirty) == "| H1 | H2 |\n|---|---|\n| a | b |"

    def test_clean_keeps_normal_table(self):
        table = "| H1 | H2 |\n|---|---|\n| a | b |"
        assert _clean_table_header(table) == table


class TestTableToNaturalLanguage:
    def test_basic_conversion(self):
        natural = table_to_natural_language("| A | B |\n|---|---|\n| 1 | 2 |")
        assert natural == "A: 1, B: 2"
        assert "|" not in natural

    def test_trailing_pipe_has_no_empty_tail(self):
        natural = table_to_natural_language("| A |\n|---|\n| x |")
        assert natural == "A: x"

    def test_mismatched_row_is_skipped(self):
        natural = table_to_natural_language(
            "| A | B |\n|---|---|\n| 1 | 2 |\n| only-one |"
        )
        assert natural == "A: 1, B: 2"

    def test_empty_value_kept(self):
        natural = table_to_natural_language("| A | B |\n|---|---|\n|  | 2 |")
        assert natural == "A: , B: 2"


class TestTableChunkerRAG:
    def test_seven_rows_two_chunks_with_header(self):
        chunks = TableChunkerRAG(chunk_size=5).chunk(_table_7(), {})
        assert len(chunks) == 2
        for chunk in chunks:
            lines = chunk.text.splitlines()
            assert lines[0] == "| A | B |"
            assert "|" in chunk.text
        assert _count_data(chunk_text(chunks[0])) == 5
        assert _count_data(chunk_text(chunks[1])) == 2

    def test_text_for_embedding_format(self):
        chunks = TableChunkerRAG(chunk_size=5).chunk(_table_7(), {})
        for chunk in chunks:
            assert chunk.text_for_embedding
            assert "|" not in chunk.text_for_embedding
            assert "A:" in chunk.text_for_embedding
            assert chunk.metadata["source_type"] == "markdown_table"

    def test_placeholder_table_is_cleaned(self):
        dirty = "|  |  |\n|---|---|\n| H1 | H2 |\n| a | b |"
        chunks = TableChunkerRAG(chunk_size=5).chunk(dirty, {})
        assert len(chunks) == 1
        assert chunks[0].text.splitlines()[0] == "| H1 | H2 |"

    def test_non_table_returns_empty(self):
        assert TableChunkerRAG().chunk("Just text", {}) == []
        assert TableChunkerRAG().chunk("", {}) == []

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError):
            TableChunkerRAG(chunk_size=0)


def chunk_text(chunk: Chunk) -> str:
    return chunk.text


def _count_data(table_markdown: str) -> int:
    lines = [line for line in table_markdown.splitlines() if line.strip()]
    return len([line for line in lines if not re.match(r"^\|[\s\-|:]+\|$", line)]) - 1


class TestHybridRagChunker:
    def _doc(self) -> str:
        table = _table_7()
        return (
            "Introductory paragraph about the report. It has several sentences. "
            "Second sentence of the intro. Third one here. " * 6
            + "\n\n"
            + table
            + "\n\n"
            + "Closing remarks of the document. One more sentence. Final words. " * 6
        )

    @pytest.mark.parametrize("strategy", ["token", "sentence", "recursive"])
    def test_mixed_document(self, strategy):
        hybrid = HybridRagChunker(
            text_strategy=strategy, chunk_size=100, overlap=10, table_chunk_size=5
        )
        chunks = hybrid.chunk(self._doc(), {"source": "doc.md"})
        assert_chunk_invariants(chunks, source="doc.md")
        kinds = [chunk.metadata["chunk_type"] for chunk in chunks]
        assert "table" in kinds and "text" in kinds
        assert [chunk.index for chunk in chunks] == list(range(len(chunks)))

        tables = [c for c in chunks if c.metadata["chunk_type"] == "table"]
        texts = [c for c in chunks if c.metadata["chunk_type"] == "text"]
        assert len(tables) == 2
        for chunk in tables:
            assert chunk.text_for_embedding
            assert "|" not in chunk.text_for_embedding
            assert chunk.metadata["source_type"] == "markdown_table"
        for chunk in texts:
            assert chunk.text_for_embedding is None

    def test_table_order_preserved(self):
        hybrid = HybridRagChunker(text_strategy="token", chunk_size=100)
        chunks = hybrid.chunk("First text block here.\n\n" + _table_7(), {})
        assert chunks[0].metadata["chunk_type"] == "text"
        assert chunks[-1].metadata["chunk_type"] == "table"

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            HybridRagChunker(text_strategy="unknown")

    def test_empty_text(self):
        assert HybridRagChunker().chunk("", {}) == []


def _crumb_tree():
    from rag_parser.core.interfaces import DocumentNode

    def node(node_type, text, children=None, level=1):
        return DocumentNode(
            type=node_type, level=level, text=text,
            children=children or [], metadata={},
        )

    return DocumentNode(
        type="document", level=0, text="", children=[
            node("header", "Раздел 2", [
                node("header", "2.3 Основные показатели", [
                    node("paragraph", "Текст про показатели."),
                    node("table", "| A | B |\n|---|---|\n| 1 | 2 |", level=0),
                ]),
            ]),
        ], metadata={},
    )


class TestHybridBreadcrumbEnrichment:
    def test_text_chunks(self):
        hybrid = HybridRagChunker(text_strategy="token", chunk_size=100, overlap=10)
        chunks = hybrid.chunk("ignored", {"source": "doc.md"}, tree=_crumb_tree())
        texts = [c for c in chunks if c.metadata["chunk_type"] == "text"]
        assert [c for c in texts if c.breadcrumb]
        for chunk in [c for c in texts if c.breadcrumb]:
            assert chunk.text_for_embedding == f"[{chunk.breadcrumb}] {chunk.text}"

    def test_table_chunks(self):
        hybrid = HybridRagChunker(text_strategy="token", chunk_size=100, overlap=10)
        chunks = hybrid.chunk("ignored", {"source": "doc.md"}, tree=_crumb_tree())
        tables = [c for c in chunks if c.metadata["chunk_type"] == "table"]
        assert tables
        for chunk in tables:
            assert chunk.breadcrumb == "Раздел 2 > 2.3 Основные показатели"
            assert chunk.text.startswith(f"[{chunk.breadcrumb}] ")
            assert chunk.text_for_embedding.startswith(f"[{chunk.breadcrumb}] ")
            assert "|" not in chunk.text_for_embedding

    def test_no_headers(self):
        from rag_parser.core.interfaces import DocumentNode

        tree = DocumentNode(
            type="document", level=0, text="", children=[
                DocumentNode(type="paragraph", level=0, text="Просто текст.",
                             children=[], metadata={}),
            ], metadata={},
        )
        hybrid = HybridRagChunker(text_strategy="token", chunk_size=100, overlap=10)
        chunks = hybrid.chunk("ignored", {}, tree=tree)
        assert all(c.breadcrumb is None for c in chunks)
