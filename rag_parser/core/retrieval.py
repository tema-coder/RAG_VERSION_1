"""Child-Parent логика выдачи при поиске (ТЗ 3.4).

Векторный поиск находит отдельный чанк. Но найденная строка таблицы или
абзац без окружения малополезны: теряется смысл раздела. Поэтому найденный
чанк разворачивается в его родителя — все чанки с тем же ``parent_id``,
упорядоченные по ``chunk_order``.

Правило из ТЗ:

* у найденного чанка есть ``parent_id`` → вернуть ВСЕ чанки этого родителя;
* ``parent_id is None`` (в документе не было заголовков) → вернуть ТОЛЬКО
  сам найденный чанк, агрегации нет.

Функция чистая: принимает список чанков, возвращает новый список. Ни
FastAPI, ни Pydantic, ни ``document_store`` здесь не используются — это
позволяет тестировать логику в изоляции и переиспользовать её в батчевой
обработке и в UI.
"""

from typing import Any, Dict, List, Sequence

from rag_parser.core.interfaces import Chunk

__all__ = [
    "expand_to_parents",
    "chunk_from_dict",
    "chunks_from_dicts",
    "group_by_parent",
]


def expand_to_parents(chunks: Sequence[Chunk], hit_index: int) -> List[Chunk]:
    """Возвращает группу найденного чанка: весь его родительский раздел.

    :param chunks: чанки документа в исходном порядке.
    :param hit_index: индекс найденного чанка в ``chunks``.
    :raises IndexError: если ``hit_index`` вне диапазона.
    """
    if hit_index < 0 or hit_index >= len(chunks):
        raise IndexError(f"chunk index out of range: {hit_index}")

    hit = chunks[hit_index]
    parent_id = getattr(hit, "parent_id", None)
    if parent_id is None:
        # Раздела нет — отдаём только сам найденный чанк (ТЗ 3.4).
        return [hit]

    # -chunk_order как tie-breaker: при равном chunk_order (теоретически
    # возможно после склейки) сохраняем исходный порядок документа.
    group = [chunk for chunk in chunks if getattr(chunk, "parent_id", None) == parent_id]
    return sorted(group, key=lambda chunk: (getattr(chunk, "chunk_order", 0), -chunk.index))


def group_by_parent(chunks: Sequence[Chunk]) -> Dict[str, List[Chunk]]:
    """Группирует чанки по ``parent_id``; чанки без родителя не попадают."""
    groups: Dict[str, List[Chunk]] = {}
    for chunk in chunks:
        parent_id = getattr(chunk, "parent_id", None)
        if parent_id is None:
            continue
        groups.setdefault(parent_id, []).append(chunk)
    for group in groups.values():
        group.sort(key=lambda chunk: (getattr(chunk, "chunk_order", 0), -chunk.index))
    return groups


def chunk_from_dict(payload: Dict[str, Any]) -> Chunk:
    """Восстанавливает ``Chunk`` из словаря ``Chunk.to_dict()``.

    Нужен потому, что ``DocumentStore`` держит чанки в сериализованном виде,
    а логика группировки работает с датаклассом. Неизвестные ключи
    игнорируются, отсутствующие берутся из значений по умолчанию.
    """
    fields = {
        "index": 0,
        "text": "",
        "tokens": 0,
        "source": "",
    }
    return Chunk(
        index=int(payload.get("index", fields["index"])),
        text=str(payload.get("text", fields["text"])),
        tokens=int(payload.get("tokens", fields["tokens"])),
        source=str(payload.get("source", fields["source"])),
        metadata=dict(payload.get("metadata") or {}),
        text_for_embedding=payload.get("text_for_embedding"),
        breadcrumb=payload.get("breadcrumb"),
        parent_id=payload.get("parent_id"),
        chunk_order=int(payload.get("chunk_order", 0)),
    )


def chunks_from_dicts(payloads: Sequence[Dict[str, Any]]) -> List[Chunk]:
    """Векторная версия :func:`chunk_from_dict`."""
    return [chunk_from_dict(item) for item in payloads]