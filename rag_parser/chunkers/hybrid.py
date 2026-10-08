import re
from typing import List, Literal, Optional

from rag_parser.chunkers.base import BaseChunker, Chunk
from rag_parser.chunkers.breadcrumb import breadcrumb_for_path, enrich_text
from rag_parser.chunkers.table import (
    _clean_table_header,
    _is_markdown_table,
    table_to_natural_language,
)
from rag_parser.core.interfaces import DocumentNode

BLOCK_SPLIT_RE = re.compile(r"\n\n+")
HEADER_LINE_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _split_block(block: str) -> List[tuple]:
    """Splits a block into ``(kind, text)`` runs of table / text lines.

    Pure blocks are returned as-is; mixed blocks are separated so that only
    real table lines reach the table chunker.
    """
    lines = [line for line in block.splitlines() if line.strip()]
    if not lines:
        return []
    if _is_markdown_table(block):
        table_lines = [line for line in lines if line.strip().startswith("|")]
        text_lines = [line for line in lines if not line.strip().startswith("|")]
        if table_lines and not text_lines:
            return [("table", block.strip())]
        parts: List[tuple] = []
        if text_lines:
            parts.append(("text", "\n".join(text_lines)))
        if table_lines:
            parts.append(("table", "\n".join(table_lines)))
        return parts
    return [("text", block.strip())]


class HybridRagChunker(BaseChunker):
    """Orchestrator: tables always go through ``TableChunkerRAG``, text goes
    through the selected strategy (``token`` / ``sentence`` / ``semantic`` /
    ``recursive``).

    Table chunks carry two representations: ``text`` (original Markdown for
    the LLM) and ``text_for_embedding`` (natural language for embeddings).

    Breadcrumb enrichment: when a (full-text) ``tree`` is passed, every
    segment is chunked with the breadcrumb of its enclosing headers;
    otherwise breadcrumbs are tracked from Markdown ``#``-headers.
    Text chunks get ``text_for_embedding = "[crumb] text"``; table chunks
    get the ``[crumb]`` prefix on BOTH representations. Without headers
    above a chunk ``breadcrumb`` stays ``None``.
    """

    name = "hybrid"

    def __init__(
        self,
        text_strategy: Literal[
            "token", "sentence", "semantic", "recursive"
        ] = "token",
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        source: str = "document",
        table_chunk_size: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> None:
        from rag_parser.chunkers.sentence import SentenceChunkerRAG
        from rag_parser.chunkers.token import TokenChunkerRAG

        if text_strategy not in ("token", "sentence", "semantic", "recursive"):
            raise ValueError(f"Unknown text strategy: {text_strategy}")
        self.text_strategy = text_strategy
        self.source = source
        if text_strategy == "token":
            self.text_chunker = TokenChunkerRAG(
                chunk_size=chunk_size, overlap=overlap, source=source
            )
        elif text_strategy == "sentence":
            self.text_chunker = SentenceChunkerRAG(
                chunk_size=chunk_size, overlap=overlap, source=source
            )
        elif text_strategy == "recursive":
            from rag_parser.chunkers.recursive import RecursiveChunker

            self.text_chunker = RecursiveChunker(
                chunk_size=chunk_size, overlap=overlap, source=source
            )
        else:
            from rag_parser.chunkers.semantic import SemanticChunkerRAG

            self.text_chunker = SemanticChunkerRAG(
                chunk_size=chunk_size,
                overlap=overlap,
                source=source,
                threshold=threshold,
            )
        from rag_parser.chunkers.table import TableChunkerRAG

        self.table_chunker = TableChunkerRAG(
            chunk_size=table_chunk_size, source=source
        )
        self.chunk_size = self.text_chunker.chunk_size
        self.overlap = self.text_chunker.overlap

    def chunk(
        self,
        text: str,
        metadata: Optional[dict] = None,
        tree: Optional[DocumentNode] = None,
    ) -> List[Chunk]:
        metadata = metadata or {}
        if tree is not None:
            return self._chunk_with_tree(text, metadata, tree)
        return self._chunk_markdown(text, metadata)

    # ------------------------------------------------------------------
    # Tree path (primary): segments in document order with exact crumbs.
    # ------------------------------------------------------------------
    def _chunk_with_tree(
        self, text: str, metadata: dict, tree: DocumentNode
    ) -> List[Chunk]:
        source = metadata.get("source", self.source)
        extra = {key: value for key, value in metadata.items() if key != "source"}
        chunks: List[Chunk] = []

        segments: List[tuple] = []  # (kind, segment_text, breadcrumb)

        def visit(node: DocumentNode, trail: List[str]) -> None:
            crumb = breadcrumb_for_path(trail)
            if node.type == "document":
                for child in node.children:
                    visit(child, trail)
            elif node.type == "header":
                if (node.text or "").strip():
                    segments.append(("text", node.text.strip(), crumb))
                child_trail = [*trail, node.text.strip()] if (node.text or "").strip() else trail
                for child in node.children:
                    visit(child, child_trail)
            elif node.type == "table":
                if (node.text or "").strip():
                    segments.append(("table", node.text.strip(), crumb))
            else:  # paragraph / list_item / raw_text
                if (node.text or "").strip():
                    segments.append(("text", node.text.strip(), crumb))
                for child in node.children:
                    visit(child, trail)

        visit(tree, [])
        if not segments:
            # Пустое дерево — откат к Markdown-пути.
            return self._chunk_markdown(text, metadata)

        for kind, segment_text, crumb in segments:
            if kind == "table":
                self._emit_table(segment_text, source, extra, chunks, crumb)
            else:
                self._emit_text(segment_text, source, extra, chunks, crumb)
        return chunks

    # ------------------------------------------------------------------
    # Markdown path (fallback): breadcrumb tracking from #-headers.
    # ------------------------------------------------------------------
    def _chunk_markdown(self, text: str, metadata: dict) -> List[Chunk]:
        metadata = metadata or {}
        if not text or not text.strip():
            return []
        source = metadata.get("source", self.source)
        extra = {key: value for key, value in metadata.items() if key != "source"}
        chunks: List[Chunk] = []
        stack: List[tuple] = []  # (level, header_text)

        for block in BLOCK_SPLIT_RE.split(text):
            if not block.strip():
                continue
            headers_in_block: List[tuple] = []
            for line in block.splitlines():
                match = HEADER_LINE_RE.match(line.strip())
                if match:
                    headers_in_block.append((len(match.group(1)), match.group(2).strip()))
            if headers_in_block and not [
                line for line in block.splitlines()
                if line.strip() and not HEADER_LINE_RE.match(line.strip())
            ]:
                # Блок только из заголовков: чанки — с крошкой предков.
                crumb = breadcrumb_for_path([item[1] for item in stack])
                for _, header_text in headers_in_block:
                    self._emit_text(header_text, source, extra, chunks, crumb)
                for level, header_text in headers_in_block:
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    stack.append((level, header_text))
                continue
            for level, header_text in headers_in_block:
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, header_text))
            crumb = breadcrumb_for_path([item[1] for item in stack])
            for kind, part in _split_block(block):
                if kind == "table":
                    self._emit_table(part, source, extra, chunks, crumb)
                else:
                    self._emit_text(part, source, extra, chunks, crumb)
        return chunks

    # ------------------------------------------------------------------
    def _emit_table(
        self,
        table_text: str,
        source: str,
        extra: dict,
        chunks: List[Chunk],
        breadcrumb: Optional[str],
    ) -> None:
        cleaned = _clean_table_header(table_text)
        for table_chunk in self.table_chunker.chunk(
            cleaned, {"source": source, **extra}
        ):
            table_chunk.metadata["chunk_type"] = "table"
            table_chunk.metadata["chunk_index"] = len(chunks)
            table_chunk.index = len(chunks)
            table_chunk.breadcrumb = breadcrumb
            if breadcrumb:
                table_chunk.metadata["breadcrumb"] = breadcrumb
                table_chunk.text = enrich_text(table_chunk.text, breadcrumb)
                if table_chunk.text_for_embedding:
                    table_chunk.text_for_embedding = enrich_text(
                        table_chunk.text_for_embedding, breadcrumb
                    )
            chunks.append(table_chunk)

    def _emit_text(
        self,
        part: str,
        source: str,
        extra: dict,
        chunks: List[Chunk],
        breadcrumb: Optional[str],
    ) -> None:
        for text_chunk in self.text_chunker.chunk(
            part, {"source": source, **extra}
        ):
            text_chunk.metadata["chunk_type"] = "text"
            text_chunk.metadata["chunk_index"] = len(chunks)
            text_chunk.index = len(chunks)
            text_chunk.breadcrumb = breadcrumb
            if breadcrumb:
                text_chunk.metadata["breadcrumb"] = breadcrumb
                text_chunk.text_for_embedding = enrich_text(
                    text_chunk.text, breadcrumb
                )
            else:
                text_chunk.text_for_embedding = None
            chunks.append(text_chunk)


__all__ = ["HybridRagChunker", "table_to_natural_language"]
