import re
from typing import List, Literal, Optional

from rag_parser.chunkers.base import BaseChunker, Chunk
from rag_parser.chunkers.table import (
    _clean_table_header,
    _is_markdown_table,
    table_to_natural_language,
)

BLOCK_SPLIT_RE = re.compile(r"\n\n+")


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
    through the selected strategy (``token`` / ``sentence`` / ``semantic``).

    Table chunks carry two representations: ``text`` (original Markdown for
    the LLM) and ``text_for_embedding`` (natural language for embeddings).
    """

    name = "hybrid"

    def __init__(
        self,
        text_strategy: Literal["token", "sentence", "semantic"] = "token",
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        source: str = "document",
        table_chunk_size: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> None:
        from rag_parser.chunkers.sentence import SentenceChunkerRAG
        from rag_parser.chunkers.token import TokenChunkerRAG

        if text_strategy not in ("token", "sentence", "semantic"):
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

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        metadata = metadata or {}
        if not text or not text.strip():
            return []
        source = metadata.get("source", self.source)
        extra = {key: value for key, value in metadata.items() if key != "source"}
        chunks: List[Chunk] = []

        def _emit_table(table_text: str) -> None:
            cleaned = _clean_table_header(table_text)
            for table_chunk in self.table_chunker.chunk(
                cleaned, {"source": source, **extra}
            ):
                table_chunk.metadata["chunk_type"] = "table"
                table_chunk.metadata["chunk_index"] = len(chunks)
                table_chunk.index = len(chunks)
                chunks.append(table_chunk)

        def _emit_text(part: str) -> None:
            for text_chunk in self.text_chunker.chunk(
                part, {"source": source, **extra}
            ):
                text_chunk.metadata["chunk_type"] = "text"
                text_chunk.metadata["chunk_index"] = len(chunks)
                text_chunk.text_for_embedding = None
                text_chunk.index = len(chunks)
                chunks.append(text_chunk)

        for block in BLOCK_SPLIT_RE.split(text):
            if not block.strip():
                continue
            for kind, part in _split_block(block):
                if kind == "table":
                    _emit_table(part)
                else:
                    _emit_text(part)
        return chunks


__all__ = ["HybridRagChunker", "table_to_natural_language"]
