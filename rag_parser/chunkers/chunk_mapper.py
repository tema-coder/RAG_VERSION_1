"""Маппер внутренний Chunk -> архитектурный Chunk (BEAR).

Применяется ПОСЛЕ чанкинга: берёт список внутренних чанков (с
``text_for_embedding``, ``parent_id``, ``breadcrumb``) и превращает в
постоянные сущности ``Chunk`` для PostgreSQL/Qdrant.

Старый внутренний ``Chunk`` (``core/interfaces.py``) НЕ удаляется и НЕ
меняется — здесь только маппинг в архитектурный тип
(``core/entities.py``).

Импорты (правка 6, 12)
----------------------
Функция подсчёта токенов — существующая ``count_tokens`` из
``chunkers/base.py`` (НЕ своя с нуля). Маркер таблицы —
``internal_chunk.metadata.get("chunk_type") == "table"`` (ставится
гибридным чанкером в ``hybrid.py``).
"""

from __future__ import annotations

import uuid
from uuid import UUID

from rag_parser.chunkers.base import count_tokens
from rag_parser.core.entities import Chunk as ArchitecturalChunk
from rag_parser.core.interfaces import Chunk as InternalChunk


def is_table(internal_chunk: InternalChunk) -> bool:
    """Маркер таблицы во внутреннем чанке.

    ``HybridRagChunker`` ставит ``metadata["chunk_type"] = "table"`` на
    табличные чанки (``hybrid.py``) — это единственный авторитетный
    маркер. Отдельная функция хелпер держит логику в одном месте и
    переиспользуется маппером и тестами.
    """
    return (internal_chunk.metadata or {}).get("chunk_type") == "table"


def map_internal_to_architectural_chunk(
    internal_chunk: InternalChunk,
    document_id: UUID,
    document_version: int,
    sequence: int,
) -> ArchitecturalChunk:
    """Внутренний чанк -> архитектурный ``Chunk`` (BEAR).

    * ``text`` — ``internal_chunk.text_for_embedding`` (обогащённый с
      breadcrumb текст для эмбеддинга). Raw-текст в metadata НЕ
      сохраняем. Если ``text_for_embedding`` пуст (чанк без breadcrumb —
      гибридный чанкер ставит ``None``), откатываемся на
      ``internal_chunk.text``: текст параграфа сам по себе уже пригоден
      для эмбеддинга, а пустой архитектурный чанк нарушал бы контракт
      ``chunk.text не пустой`` (Шаг 7).
    * ``parent_id`` — ``internal_chunk.parent_id`` (id заголовка, ``"h_1"``)
      либо ``f"doc_{document_id}"`` для чанков без заголовка-родителя
      (группирует «бессемейные» чанки документа в одну логическую группу
      для Parent-Child retrieval).
    * ``token_count`` — ``count_tokens(text)`` (существующая функция).
    * ``metadata``:
        - ``block_type``: ``"table"``/``"text"`` (новая конвенция BEAR;
          отличается от внутреннего ``chunk_type``, но выводится из него);
        - ``parent_id``: для Parent-Child retrieval (см. docs);
        - ``breadcrumb``: путь заголовков до чанка;
        - ``mime_type``: ``None`` (заполняется позже в ingestion pipeline);
        - ``table_representation``: ``"markdown"`` для таблиц, иначе ``None``;
        - ``attachments``: ``[]`` (заглушка).
    """
    # text_for_embedding — обогащённый текст (с breadcrumb). У чанков без
    # breadcrumb гибридный чанкер оставляет None — откатываемся на raw-текст.
    text = internal_chunk.text_for_embedding or internal_chunk.text
    parent_id = getattr(internal_chunk, "parent_id", None) or f"doc_{document_id}"
    block_type = "table" if is_table(internal_chunk) else "text"

    return ArchitecturalChunk(
        id=uuid.uuid4(),
        document_id=document_id,
        document_version=document_version,
        sequence=sequence,
        text=text,
        token_count=count_tokens(text),
        metadata={
            "block_type": block_type,
            "parent_id": parent_id,  # Ключевое поле для Parent-Child retrieval
            "breadcrumb": getattr(internal_chunk, "breadcrumb", None),
            "mime_type": None,  # Заполнится позже в ingestion pipeline
            "table_representation": "markdown" if block_type == "table" else None,
            "attachments": [],  # Заглушка
        },
    )


__all__ = ["is_table", "map_internal_to_architectural_chunk"]
