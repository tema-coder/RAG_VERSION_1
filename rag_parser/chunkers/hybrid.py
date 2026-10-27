import itertools
import re
from typing import Dict, List, Literal, Optional

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
    Every table chunk holds exactly ONE data row plus the table header
    (ТЗ 3.3) — the row count is not configurable.

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
        text_strategy: Literal["token", "sentence", "semantic", "recursive"] = "token",
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        source: str = "document",
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

        # Размер табличного чанка НЕ настраивается: ТЗ 3.3 требует
        # ровно одну строку данных на чанк.
        self.table_chunker = TableChunkerRAG(source=source)
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

    def chunk_architectural(
        self,
        parsed_document,
        document_version: int = 1,
    ):
        """Архитектурный чанкинг BEAR для гибридного чанкера.

        НОВЫЙ метод — НЕ заменяет ``.chunk()`` (старая сигнатура с
        ``tree=`` остаётся для API, ``process_input_folder`` и тестов).

        Согласно Шагу 5 промпта, внутренние чанки получаем передачей
        ``parsed_document.text`` (markdown) в старую логику ``.chunk()``
        — без ``tree=``. Breadcrumb при этом восстанавливается из
        markdown ``#``-заголовков (метод ``_chunk_markdown``), что даёт
        корректные ``text_for_embedding`` и ``parent_id`` для маппинга.

        Затем каждый внутренний чанк маппится в архитектурный через
        ``map_internal_to_architectural_chunk`` с глобальным ``sequence``.
        """
        from rag_parser.chunkers.chunk_mapper import (
            map_internal_to_architectural_chunk,
        )

        internal_chunks = self.chunk(parsed_document.text, {"source": "document"})
        return [
            map_internal_to_architectural_chunk(
                internal_chunk=chunk,
                document_id=parsed_document.document_id,
                document_version=document_version,
                sequence=i,
            )
            for i, chunk in enumerate(internal_chunks)
        ]

    # ------------------------------------------------------------------
    # Tree path (primary): segments in document order with exact crumbs.
    # ------------------------------------------------------------------
    def _chunk_with_tree(
        self, text: str, metadata: dict, tree: DocumentNode
    ) -> List[Chunk]:
        source = metadata.get("source", self.source)
        extra = {key: value for key, value in metadata.items() if key != "source"}
        chunks: List[Chunk] = []

        segments: List[tuple] = []  # (kind, segment_text, crumb, parent_id)

        # Детерминированные id заголовков: h_1, h_2, ... по порядку обхода.
        # Порядковый номер не зависит от текста заголовка, поэтому повторный
        # разбор того же документа даёт те же id (проверяется тестом).
        header_counter = itertools.count(1)
        # (id, уровень) — стек открытых заголовков; верхний = ближайший родитель.
        header_stack: List[tuple] = []

        def visit(node: DocumentNode, trail: List[str]) -> None:
            crumb = breadcrumb_for_path(trail)
            if node.type == "document":
                for child in node.children:
                    visit(child, trail)
            elif node.type == "header":
                title = (node.text or "").strip()
                if title:
                    header_id = f"h_{next(header_counter)}"
                    # Заголовок входит в свой собственный раздел: его текст —
                    # первый чанк этого родителя.
                    segments.append(("text", title, crumb, header_id))
                    # Уровень узла задаёт вложенность: снимаем более глубокие.
                    level = (
                        node.level
                        if isinstance(node.level, int)
                        else len(header_stack) + 1
                    )
                    while header_stack and header_stack[-1][1] >= level:
                        header_stack.pop()
                    header_stack.append((header_id, level))
                    for child in node.children:
                        visit(child, [*trail, title])
                    return
                # Заголовок без текста — не становится родителем.
                for child in node.children:
                    visit(child, trail)
            elif node.type == "table":
                if (node.text or "").strip():
                    segments.append(
                        ("table", node.text.strip(), crumb, current_parent())
                    )
            else:  # paragraph / list_item / raw_text
                if (node.text or "").strip():
                    segments.append(
                        ("text", node.text.strip(), crumb, current_parent())
                    )
                for child in node.children:
                    visit(child, trail)

        def current_parent() -> Optional[str]:
            return header_stack[-1][0] if header_stack else None

        visit(tree, [])
        if not segments:
            # Пустое дерево — откат к Markdown-пути.
            return self._chunk_markdown(text, metadata)

        # Порядковый номер внутри родителя: счётчик перезапускается при
        # смене parent_id, поэтому у каждого заголовка нумерация с 0.
        # Важно: один сегмент может дать НЕСКОЛЬКО чанков (таблица режется
        # по одной строке), поэтому счётчик двигается на число реально
        # созданных чанков, а не на один.
        orders: Dict[str, int] = {}
        for kind, segment_text, crumb, parent_id in segments:
            before = len(chunks)
            if kind == "table":
                self._emit_table(
                    segment_text,
                    source,
                    extra,
                    chunks,
                    crumb,
                    parent_id,
                    orders.get(parent_id, 0) if parent_id is not None else 0,
                )
            else:
                self._emit_text(
                    segment_text,
                    source,
                    extra,
                    chunks,
                    crumb,
                    parent_id,
                    orders.get(parent_id, 0) if parent_id is not None else 0,
                )
            if parent_id is not None:
                orders[parent_id] = orders.get(parent_id, 0) + (len(chunks) - before)
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
        header_counter = itertools.count(1)
        parent_stack: List[tuple] = []  # (id, level) — как в tree-пути
        orders: Dict[str, int] = {}

        def next_order(parent_id: Optional[str]) -> int:
            if parent_id is None:
                return 0
            order = orders.get(parent_id, 0)
            orders[parent_id] = order + 1
            return order

        def next_order_after(parent_id: Optional[str], produced: int) -> None:
            """Сдвинуть счётчик на число реально созданных чанков сегмента."""
            if parent_id is not None:
                orders[parent_id] = orders.get(parent_id, 0) + produced

        for block in BLOCK_SPLIT_RE.split(text):
            if not block.strip():
                continue
            headers_in_block: List[tuple] = []
            for line in block.splitlines():
                match = HEADER_LINE_RE.match(line.strip())
                if match:
                    headers_in_block.append(
                        (len(match.group(1)), match.group(2).strip())
                    )
            if headers_in_block and not [
                line
                for line in block.splitlines()
                if line.strip() and not HEADER_LINE_RE.match(line.strip())
            ]:
                # Блок только из заголовков: чанки — с крошкой предков.
                crumb = breadcrumb_for_path([item[1] for item in stack])
                for level, header_text in headers_in_block:
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    while parent_stack and parent_stack[-1][1] >= level:
                        parent_stack.pop()
                    stack.append((level, header_text))
                    header_id = f"h_{next(header_counter)}"
                    parent_stack.append((header_id, level))
                    # Заголовок — первый чанк своего же раздела.
                    self._emit_text(
                        header_text,
                        source,
                        extra,
                        chunks,
                        crumb,
                        header_id,
                        next_order(header_id),
                    )
                continue
            for level, header_text in headers_in_block:
                while stack and stack[-1][0] >= level:
                    stack.pop()
                while parent_stack and parent_stack[-1][1] >= level:
                    parent_stack.pop()
                stack.append((level, header_text))
                parent_stack.append((f"h_{next(header_counter)}", level))
            crumb = breadcrumb_for_path([item[1] for item in stack])
            parent_id = parent_stack[-1][0] if parent_stack else None
            for kind, part in _split_block(block):
                before = len(chunks)
                if kind == "table":
                    self._emit_table(
                        part,
                        source,
                        extra,
                        chunks,
                        crumb,
                        parent_id,
                        orders.get(parent_id, 0) if parent_id else 0,
                    )
                else:
                    self._emit_text(
                        part,
                        source,
                        extra,
                        chunks,
                        crumb,
                        parent_id,
                        orders.get(parent_id, 0) if parent_id else 0,
                    )
                next_order_after(parent_id, len(chunks) - before)
        return chunks

    # ------------------------------------------------------------------
    def _emit_table(
        self,
        table_text: str,
        source: str,
        extra: dict,
        chunks: List[Chunk],
        breadcrumb: Optional[str],
        parent_id: Optional[str] = None,
        chunk_order: int = 0,
    ) -> None:
        cleaned = _clean_table_header(table_text)
        # Строки одной таблицы идут подряд и принадлежат одному родителю,
        # поэтому нумерация продолжается, а не перезапускается с 0.
        order = chunk_order
        for table_chunk in self.table_chunker.chunk(
            cleaned, {"source": source, **extra}
        ):
            table_chunk.metadata["chunk_type"] = "table"
            table_chunk.metadata["chunk_index"] = len(chunks)
            table_chunk.index = len(chunks)
            table_chunk.breadcrumb = breadcrumb
            table_chunk.parent_id = parent_id
            table_chunk.chunk_order = order
            if parent_id is not None:
                table_chunk.metadata["parent_id"] = parent_id
                table_chunk.metadata["chunk_order"] = order
            order += 1
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
        parent_id: Optional[str] = None,
        chunk_order: int = 0,
    ) -> None:
        order = chunk_order
        for text_chunk in self.text_chunker.chunk(part, {"source": source, **extra}):
            text_chunk.metadata["chunk_type"] = "text"
            text_chunk.metadata["chunk_index"] = len(chunks)
            text_chunk.index = len(chunks)
            text_chunk.breadcrumb = breadcrumb
            text_chunk.parent_id = parent_id
            text_chunk.chunk_order = order
            if parent_id is not None:
                text_chunk.metadata["parent_id"] = parent_id
                text_chunk.metadata["chunk_order"] = order
            order += 1
            if breadcrumb:
                text_chunk.metadata["breadcrumb"] = breadcrumb
                text_chunk.text_for_embedding = enrich_text(text_chunk.text, breadcrumb)
            else:
                text_chunk.text_for_embedding = None
            chunks.append(text_chunk)


__all__ = ["HybridRagChunker", "table_to_natural_language"]
