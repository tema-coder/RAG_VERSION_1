"""Архитектурные сущности BEAR (целевая архитектура).

Тонкий слой постоянных/временных сущностей, в которые маппятся внутренние
структуры парсера и чанкера перед сохранением в PostgreSQL/Qdrant.

Это НЕ замена внутренних ``ParseResult``/``Chunk`` из ``core/interfaces.py``:
старые структуры остаются рабочими, а здесь лежат параллельные
архитектурные типы. Мапперы-адаптеры (``parsers/result_mapper.py``,
``chunkers/chunk_mapper.py``) преобразуют одно в другое.

Изображения (``ContentBlockType.IMAGE``) из текущего дерева ``DocumentNode``
получить нельзя — dedoc не отдаёт изображения как узлы. Enum добавлен про
запас: когда появится image-extractor, тип будет готов.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID


class ContentBlockType(Enum):
    """Тип блока контента в ParsedDocument."""

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


@dataclass
class ContentBlock:
    """Блок контента документа (плоский список, без вложенности).

    ``content``:
      * ``str`` — для TEXT (текст параграфа/пункта списка);
      * ``str`` (markdown-таблица) — для TABLE (на этапе парсинга);
      * ``bytes`` — для IMAGE (когда появится image-extractor).

    ``position`` — порядковый номер блока в документе (0-based), нужен
    для восстановления порядка при сохранении и поиске.

    ``metadata`` — breadcrumb пути заголовков, исходный тип узла, уровень
    и прочее; ``None`` если метаданных нет.
    """

    type: ContentBlockType
    content: Any  # str для TEXT, dict/строка для TABLE, bytes для IMAGE
    position: int
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ParsedDocument:
    """Временная сущность: результат парсинга, вход для чанкинга.

    ``text`` — весь извлечённый текст документа (markdown), используется
    чанкером как источник для разбиения.

    ``blocks`` — плоский список блоков контента в порядке документа;
    заголовки сюда НЕ входят (они несут breadcrumb-роль, а не контент).

    ``attachments`` — заглушка ``[]``: вложения (изображения и пр.) на
    этом этапе не извлекаются, поле добавлено под будущий ingestion.
    """

    document_id: UUID
    text: str  # весь извлечённый текст (markdown)
    blocks: List[ContentBlock]
    attachments: List[Dict[str, Any]]  # ПОКА ЗАГЛУШКА: []
    metadata: Dict[str, Any]


@dataclass
class Chunk:
    """Постоянная сущность чанка (сохраняется в PostgreSQL/Qdrant).

    ``text`` — ТОЛЬКО обогащённый текст для эмбеддинга (маппится из
    внутреннего ``Chunk.text_for_embedding``). Raw-текст сюда НЕ
    кладётся и в metadata НЕ дублируется.

    ``metadata`` содержит ключи для Parent-Child retrieval:
      * ``block_type`` — ``"text"``/``"table"`` (новая конвенция BEAR);
      * ``parent_id`` — id родительского заголовка (``"h_1"``) либо
        ``f"doc_{document_id}"`` для чанков без заголовка-родителя;
      * ``breadcrumb`` — путь заголовков до чанка;
      * ``mime_type`` — ``None`` (заполняется позже в ingestion pipeline);
      * ``table_representation`` — ``"markdown"`` для таблиц, иначе ``None``;
      * ``attachments`` — ``[]`` (заглушка).
    """

    id: UUID
    document_id: UUID
    document_version: int
    sequence: int
    text: str  # Сюда маппим internal_chunk.text_for_embedding
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "ContentBlock",
    "ContentBlockType",
    "ParsedDocument",
    "Chunk",
]
