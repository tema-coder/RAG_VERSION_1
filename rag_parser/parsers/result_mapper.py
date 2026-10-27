"""Маппер ParseResult -> ParsedDocument (архитектура BEAR).

Тонкий слой-адаптер: преобразует внутреннюю структуру парсера
(``ParseResult`` с ``DocumentNode``-деревом) во временную сущность
``ParsedDocument``, которую потребляет чанкинг.

Старый ``ParseResult`` НЕ удаляется и НЕ меняется — маппер вызывается
отдельно (см. ``DedocParser.parse_to_document`` или напрямую).

Алгоритм обхода дерева
----------------------
Рекурсивный pre-order обход ``parse_result.tree``. Правила маппинга узлов
``DocumentNode`` в ``ContentBlock``:

* ``document`` — контейнер, обходим детей, блок не создаём;
* ``header`` — НЕ создаёт отдельный блок: его текст накапливается в
  breadcrumb-trail и становится контекстом последующих TEXT/TABLE-блоков
  (заголовки нужны для breadcrumb, а не как самостоятельный контент);
* ``table`` → ``ContentBlockType.TABLE`` (контент — markdown-текст таблицы);
* ``paragraph``/``list_item``/``raw_text`` → ``ContentBlockType.TEXT``;
* ``IMAGE`` из текущего дерева получить нельзя (dedoc не отдаёт изображения
  как узлы) — ``ContentBlockType.IMAGE`` на этом этапе не порождается.

Breadcrumb берётся из пути заголовков строго ДО текущего узла (сам
заголовок в свою крошку не входит), через существующий хелпер
``breadcrumb_for_path`` из ``chunkers/breadcrumb.py``.
"""

from __future__ import annotations

from typing import List
from uuid import UUID

from rag_parser.chunkers.breadcrumb import breadcrumb_for_path
from rag_parser.core.entities import (
    ContentBlock,
    ContentBlockType,
    ParsedDocument,
)
from rag_parser.core.interfaces import DocumentNode, ParseResult


def _block_type_for(node: DocumentNode) -> ContentBlockType:
    """Тип ContentBlock по типу узла DocumentNode."""
    if node.type == "table":
        return ContentBlockType.TABLE
    # paragraph / list_item / raw_text / прочее — текстовый блок.
    return ContentBlockType.TEXT


def _walk_for_blocks(
    node: DocumentNode,
    trail: List[str],
    blocks: List[ContentBlock],
) -> None:
    """Рекурсивный обход дерева в ContentBlock (pre-order).

    ``trail`` — путь заголовков от корня строго до текущего узла. Для
    заголовка trail детей = trail + его текст; для TEXT/TABLE-узла
    текущий trail и есть его breadcrumb.
    """
    if node.type == "document":
        for child in node.children:
            _walk_for_blocks(child, trail, blocks)
        return

    if node.type == "header":
        # Заголовок НЕ создаёт блок — он несёт breadcrumb-роль.
        title = (node.text or "").strip()
        child_trail = [*trail, title] if title else trail
        for child in node.children:
            _walk_for_blocks(child, child_trail, blocks)
        return

    # TEXT / TABLE лист (после пре-сплита promoter-а — обычно однострочный).
    text = (node.text or "").strip()
    if not text:
        # Пустой узел — блок не создаём, но детей обходим (вдруг есть).
        for child in node.children:
            _walk_for_blocks(child, trail, blocks)
        return

    breadcrumb = breadcrumb_for_path(trail)
    block = ContentBlock(
        type=_block_type_for(node),
        content=text,
        position=len(blocks),
        metadata={
            "breadcrumb": breadcrumb,
            "node_type": node.type,
            "level": node.level,
            # Копируем метаданные узла, отбрасывая сырые аннотации
            # (промоутеру они нужны были, в архитектурной сущности — лишний груз).
            "node_metadata": {
                key: value
                for key, value in node.metadata.items()
                if key != "annotations"
            },
        },
    )
    blocks.append(block)
    # Таблицы/параграфы в текущем дереве детей-блоков не имеют, но на
    # всякий случай обходим — вложенные узлы станут отдельными блоками.
    for child in node.children:
        _walk_for_blocks(child, trail, blocks)


def map_parse_result_to_parsed_document(
    parse_result: ParseResult,
) -> ParsedDocument:
    """``ParseResult`` (внутренняя структура) -> ``ParsedDocument`` (BEAR).

    * ``document_id``: ``UUID(parse_result.document_id)`` — внутренний id
      это hex-строка (``uuid4().hex``), корректно парсится ``UUID()``;
    * ``text``: ``parse_result.markdown`` (весь извлечённый текст);
    * ``blocks``: обход ``parse_result.tree`` (без заголовков-блоков);
    * ``attachments``: ``[]`` (заглушка);
    * ``metadata``: копия ``parse_result.metadata``.
    """
    blocks: List[ContentBlock] = []
    if parse_result.tree is not None:
        _walk_for_blocks(parse_result.tree, [], blocks)

    return ParsedDocument(
        document_id=UUID(parse_result.document_id),
        text=parse_result.markdown,
        blocks=blocks,
        attachments=[],
        metadata=dict(parse_result.metadata),
    )


__all__ = ["map_parse_result_to_parsed_document"]
