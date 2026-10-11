"""Хлебные крошки: путь заголовков от корня до родителя узла.

Формат: ``"Раздел 2 > 2.3 Основные показатели"``, разделитель ``" > "``.

Глубина ограничена ``BREADCRUMB_MAX_DEPTH`` из конфига (по умолчанию **2**) —
берутся ПОСЛЕДНИЕ N уровней пути, ближайшие к чанку несут больше смысла для
эмбеддинга. Примеры (ТЗ 3.2):

* ``1.1`` -> ``1.1.1``           -> ``"1.1 Название > 1.1.1 Название"``
* ``1.1`` -> ``1.1.1`` -> ``1.1.1.1`` -> ``"1.1.1 Название > 1.1.1.1 Название"``

Если путь короче лимита — берётся весь. Без заголовков над чанком —
``None``, обогащение не применяется. ``max_depth <= 0`` отключает
ограничение (в крошку идёт весь путь) — нужен для отладки.
"""

from typing import List, Optional

from rag_parser.core.interfaces import DocumentNode

SEPARATOR = " > "


def _full_text(node: DocumentNode) -> str:
    """Полный текст узла (до урезания превью).

    Узлы полного дерева хранят полный текст; у узлов превью-дерева текст
    уже урезан до 30 символов — для breadcrumb годится любой: крошка
    строится из заголовков, короткие заголовки урезанием не страдают.
    """
    return (node.text or "").strip()


def breadcrumb_for_path(headers: List[str], max_depth: Optional[int] = None) -> Optional[str]:
    """Строит крошку из списка заголовков (корень → родитель)."""
    from rag_parser.core.config import settings

    limit = int(settings.BREADCRUMB_MAX_DEPTH if max_depth is None else max_depth)
    trail = [part.strip() for part in headers if part and part.strip()]
    if not trail:
        return None
    if limit > 0:
        trail = trail[-limit:]
    return SEPARATOR.join(trail)


def _walk(
    node: DocumentNode,
    trail: List[str],
    out: List[tuple],
) -> None:
    """Собирает ``(узел, крошка)`` для каждого узла дерева.

    Крошка узла — путь заголовков от корня строго ДО узла (сам заголовок
    в свою крошку не входит, только предки).
    """
    out.append((node, breadcrumb_for_path(trail)))
    if node.type == "header":
        text = _full_text(node)
        if text:
            trail = [*trail, text]
    for child in node.children:
        _walk(child, trail, out)


def annotate_tree(root: DocumentNode) -> List[tuple]:
    """Возвращает ``[(узел, breadcrumb)]`` для всех узлов дерева."""
    out: List[tuple] = []
    _walk(root, [], out)
    return out


def enrich_text(text: str, breadcrumb: Optional[str]) -> str:
    """``[breadcrumb] текст`` либо исходный текст без крошки."""
    if breadcrumb:
        return f"[{breadcrumb}] {text}"
    return text


__all__ = [
    "SEPARATOR",
    "annotate_tree",
    "breadcrumb_for_path",
    "enrich_text",
]
