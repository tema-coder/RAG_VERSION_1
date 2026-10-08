"""Нормализация сырого вывода dedoc в :class:`DocumentNode`.

Учитывает выверенные грабли API dedoc:

* текст строки — ``line.line``, НЕ ``line.text``; значение может быть
  вложенным ``LineWithMeta`` — раскрывается рекурсивно; в API-схеме
  (``model_dump``) то же поле называется ``text`` — поддерживаются оба;
* тип и уровень — ``line.metadata.data["paragraph_type"]`` и
  ``line.metadata.data["hierarchy_level"]`` (объект/словарь с полями
  ``level_1``, ``level_2``, ``line_type``); атрибутов ``line.type`` /
  ``line.level`` не существует; в API-схеме ``metadata`` — плоский словарь
  с ключом ``paragraph_type`` — поддерживается тоже;
* заголовки называются ``"header"``, не ``"heading"`` (значение
  ``"heading"`` молча нормализуется);
* ячейка таблицы — ``CellWithMeta``, у неё нет атрибута ``.text``; текст
  достаётся через ``cell.get_text()`` либо обходом ``cell.lines``;
  строковое представление ячейки даёт мусор вида
  ``CellWithMeta(cs=1, rs=1, ...)`` — полагаться на него запрещено;
* аннотации целиком не копируются — только ``annotations_count``.

Урезание ``text`` до ``TREE_TEXT_PREVIEW_LEN`` + ``"…"`` — последний шаг
(:func:`truncate`), после генерации Markdown и передачи полных текстов
в чанкер. :func:`build_full_tree` возвращает дерево с ПОЛНЫМИ текстами.
"""

from typing import Any, Dict, List, Optional, Tuple

from rag_parser.core.interfaces import DocumentNode

ALLOWED_TYPES = ("document", "header", "paragraph", "list_item", "table", "raw_text")

_HEADER_ALIASES = {"header", "heading", "h1", "h2", "h3", "h4", "h5", "h6", "title"}
_LIST_ALIASES = {"list_item", "list", "bullet_list_item", "ordered_list_item"}


def _is_header_type(paragraph_type: str) -> bool:
    return str(paragraph_type or "").lower() in _HEADER_ALIASES


def _normalize_type(paragraph_type: str, has_text: bool) -> str:
    lowered = str(paragraph_type or "").lower()
    if lowered in _HEADER_ALIASES:
        return "header"
    if lowered in _LIST_ALIASES:
        return "list_item"
    if lowered == "table":
        return "table"
    if lowered == "raw_text":
        return "raw_text"
    if lowered in ("root", "toc", "document"):
        return "document"
    return "paragraph" if has_text else "raw_text"


def extract_line_text(line: Any) -> str:
    """Достаёт текст из строки dedoc в любом представлении.

    Поддерживает: ``str``, словарь API-схемы (``text``) и словарь
    внутренней структуры (``line`` — возможно вложенный), объекты
    ``LineWithMeta`` (``.line`` — возможно вложенный ``LineWithMeta``,
    ``.text`` у API-объектов), объекты с ``get_text()``.
    """
    if line is None:
        return ""
    if isinstance(line, str):
        return line
    if isinstance(line, dict):
        if "line" in line and line["line"] is not None:
            nested = extract_line_text(line["line"])
            if nested:
                return nested
        if "text" in line and isinstance(line["text"], str):
            return line["text"]
        if "text" in line and line["text"] is not None:
            nested = extract_line_text(line["text"])
            if nested:
                return nested
        return ""
    get_text = getattr(line, "get_text", None)
    if callable(get_text):
        try:
            text = get_text()
        except Exception:  # noqa: BLE001 - defensive
            text = ""
        if isinstance(text, str) and text:
            return text
    if hasattr(line, "line"):
        try:
            return extract_line_text(line.line)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - defensive
            pass
    if hasattr(line, "text"):
        try:
            return extract_line_text(line.text)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - defensive
            pass
    if isinstance(line, (list, tuple)):
        return "\n".join(part for part in (extract_line_text(p) for p in line) if part)
    result = str(line)
    if result.startswith(("CellWithMeta(", "LineWithMeta(")):
        return ""
    return result


def _as_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    data = getattr(obj, "data", None)
    if isinstance(data, dict):
        return data
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        try:
            dumped = obj.model_dump()
        except Exception:  # noqa: BLE001 - defensive
            dumped = {}
        if isinstance(dumped, dict):
            return dumped
    result: Dict[str, Any] = {}
    for key in ("paragraph_type", "hierarchy_level", "page_id", "line_id"):
        if hasattr(obj, key):
            try:
                result[key] = getattr(obj, key)
            except Exception:  # noqa: BLE001 - defensive
                continue
    return result


def extract_paragraph_info(metadata: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Возвращает ``(paragraph_type, hierarchy_level)`` из метаданных узла.

    Метаданные могут быть: плоским словарём API-схемы
    (``{"paragraph_type": ...}``), словарём с вложенным ``"data"``,
    объектом ``LineMetadata`` (``.hierarchy_level`` / ``.tag_hierarchy_level``)
    либо pydantic-моделью API-схемы.
    """
    if metadata is None:
        return "raw_text", None
    meta = _as_dict(metadata)
    data = meta.get("data")
    if isinstance(data, dict):
        meta = {**meta, **data}
    paragraph_type = meta.get("paragraph_type") or ""
    hierarchy = meta.get("hierarchy_level") or meta.get("hierarchyLevel")
    if hierarchy is not None and not isinstance(hierarchy, dict):
        level_1 = getattr(hierarchy, "level_1", None)
        level_2 = getattr(hierarchy, "level_2", None)
        line_type = getattr(hierarchy, "line_type", None)
        hierarchy = {"level_1": level_1, "level_2": level_2, "line_type": line_type}
    if not paragraph_type and isinstance(hierarchy, dict):
        paragraph_type = hierarchy.get("line_type") or ""
    return str(paragraph_type or "raw_text"), hierarchy


def extract_annotations_count(node: Any) -> int:
    if isinstance(node, dict):
        annotations = node.get("annotations")
    else:
        annotations = getattr(node, "annotations", None)
    if annotations is None:
        return 0
    try:
        return len(annotations)
    except TypeError:
        return 0


def header_level_from_hierarchy(hierarchy: Optional[Dict[str, Any]]) -> Optional[int]:
    """Уровень заголовка из ``hierarchy_level`` (level_2 приоритетнее)."""
    if not hierarchy:
        return None
    for key in ("level_2", "level_1"):
        value = hierarchy.get(key)
        if isinstance(value, int) and value >= 1:
            return value
    return None


def _subparagraphs_of(node: Any) -> List[Any]:
    if isinstance(node, dict):
        for key in ("subparagraphs", "sub_paragraphs", "children"):
            value = node.get(key)
            if isinstance(value, list):
                return value
        return []
    for attr in ("subparagraphs", "sub_paragraphs", "children"):
        value = getattr(node, attr, None)
        if isinstance(value, list):
            return value
    return []


def _node_text_of(node: Any) -> str:
    if isinstance(node, dict):
        for key in ("text", "line"):
            if key in node:
                text = extract_line_text(node[key])
                if text:
                    return text
        return extract_line_text(node)
    for attr in ("text", "line"):
        if hasattr(node, attr):
            try:
                text = extract_line_text(getattr(node, attr))
            except Exception:  # noqa: BLE001 - defensive
                continue
            if text:
                return text
    return ""


def _node_meta_of(node: Any) -> Any:
    if isinstance(node, dict):
        return node.get("metadata")
    return getattr(node, "metadata", None)


def cell_to_text(cell: Any) -> str:
    """Текст ячейки таблицы: ``get_text()`` либо обход ``lines``."""
    if cell is None:
        return ""
    if isinstance(cell, str):
        return cell
    if isinstance(cell, dict):
        lines = cell.get("lines")
        if isinstance(lines, list):
            return "\n".join(
                part for part in (extract_line_text(ln) for ln in lines) if part
            )
        return extract_line_text(cell)
    get_text = getattr(cell, "get_text", None)
    if callable(get_text):
        try:
            text = get_text()
        except Exception:  # noqa: BLE001 - defensive
            text = ""
        if isinstance(text, str) and text:
            return text
    lines = getattr(cell, "lines", None)
    if isinstance(lines, list):
        return "\n".join(
            part for part in (extract_line_text(ln) for ln in lines) if part
        )
    return ""


def _table_cells_of(table: Any) -> List[List[Any]]:
    if isinstance(table, dict):
        cells = table.get("cells")
    else:
        cells = getattr(table, "cells", None)
    if isinstance(cells, list):
        return cells
    return []


def table_to_rows(table: Any) -> List[List[str]]:
    """Таблица dedoc → матрица строк (тексты через :func:`cell_to_text`)."""
    rows: List[List[str]] = []
    for row in _table_cells_of(table):
        if not isinstance(row, (list, tuple)):
            continue
        rows.append([cell_to_text(cell) for cell in row])
    return rows


def _build_children(
    raw_children: List[Any], parent_level: int, header_depth: int
) -> List[DocumentNode]:
    return [
        _convert_node(child, parent_level=parent_level, header_depth=header_depth)
        for child in raw_children
    ]


def _raw_annotations_of(node: Any) -> List[Dict[str, Any]]:
    if isinstance(node, dict):
        annotations = node.get("annotations")
    else:
        annotations = getattr(node, "annotations", None)
    if not isinstance(annotations, list):
        return []
    result: List[Dict[str, Any]] = []
    for item in annotations:
        if isinstance(item, dict):
            result.append(
                {
                    "name": item.get("name", ""),
                    "start": item.get("start", 0),
                    "end": item.get("end", 0),
                    "value": item.get("value", ""),
                }
            )
        else:
            result.append(
                {
                    "name": getattr(item, "name", ""),
                    "start": getattr(item, "start", 0),
                    "end": getattr(item, "end", 0),
                    "value": getattr(item, "value", ""),
                }
            )
    return result


def _convert_node(node: Any, parent_level: int, header_depth: int) -> DocumentNode:
    text = _node_text_of(node).strip()
    paragraph_type, hierarchy = extract_paragraph_info(_node_meta_of(node))
    node_type = _normalize_type(paragraph_type, has_text=bool(text))
    level = parent_level
    current_depth = header_depth
    if node_type == "header":
        from_hierarchy = header_level_from_hierarchy(hierarchy)
        level = from_hierarchy if from_hierarchy else header_depth + 1
        current_depth = level
    children = _build_children(
        _subparagraphs_of(node), parent_level=level, header_depth=current_depth
    )
    raw_annotations = _raw_annotations_of(node)
    return DocumentNode(
        type=node_type,
        level=level,
        text=text,
        children=children,
        metadata={
            "annotations_count": len(raw_annotations),
            "annotations": raw_annotations,
        },
    )


def build_full_tree(structure: Any, tables: Optional[List[Any]] = None) -> DocumentNode:
    """Строит дерево с ПОЛНЫМИ текстами из структуры dedoc и списка таблиц.

    ``structure`` — словарь/объект узла ``content.structure``
    (``model_dump`` API-схемы) либо уже список узлов. Таблицы из
    ``content.tables`` привязываются к узлам-якорям по ``TableAnnotation``
    (первый ребёнок якоря); без привязки — append в конец корня.
    Для боевого пайплайна используйте `reattach_tables` после промоции.
    Возвращает ``(root, orphan_tables)`` через metadata корня.
    """
    if isinstance(structure, list):
        children = _build_children(structure, parent_level=0, header_depth=0)
        root_text = ""
        root_annotations: List[Dict[str, Any]] = []
    else:
        text = _node_text_of(structure).strip()
        paragraph_type, _ = extract_paragraph_info(_node_meta_of(structure))
        root_type = _normalize_type(paragraph_type, has_text=bool(text))
        if root_type != "document":
            root_type = "document"
        children = _build_children(
            _subparagraphs_of(structure), parent_level=0, header_depth=0
        )
        root_text = text
        root_annotations = _raw_annotations_of(structure)
    orphan_tables = _attach_tables(children, tables)
    return DocumentNode(
        type="document",
        level=0,
        text=root_text,
        children=children,
        metadata={
            "annotations_count": len(root_annotations),
            "annotations": root_annotations,
            "orphan_tables": orphan_tables,
        },
    )


def _table_uid(table: Any) -> Optional[str]:
    meta = (
        table.get("metadata")
        if isinstance(table, dict)
        else getattr(table, "metadata", None)
    )
    if isinstance(meta, dict):
        uid = meta.get("uid")
    else:
        uid = getattr(meta, "uid", None)
    return str(uid) if uid else None


def _anchor_uids(raw_annotations: List[Dict[str, Any]]) -> List[str]:
    """Значения `TableAnnotation` (`name == "table"`) узла-якоря."""
    uids: List[str] = []
    for item in raw_annotations or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("name", "")).lower() == "table" and item.get("value"):
            uids.append(str(item["value"]))
    return uids


def _table_node(md_table: str) -> DocumentNode:
    return DocumentNode(
        type="table",
        level=0,
        text=md_table,
        children=[],
        metadata={"annotations_count": 0, "annotations": []},
    )


def _attach_tables(
    children: List[DocumentNode], tables: List[Any]
) -> int:
    """Привязка таблиц к узлам-якорям по `TableAnnotation`.

    Таблица вставляется первым ребёнком якоря (позиция dedoc: конец текста
    узла, до остальных детей). Таблицы без якоря (нет аннотации, неизвестный
    uid, дубли, Excel/CSV без привязки) — append в конец + warning.
    Возвращает число orphan-таблиц.

    NB: вызывать ПОСЛЕ промоции (`promote_headers`): привязка до неё делает
    якорь composite-узлом и блокирует резку многострочников — merged-узел
    «Раздел 6 + тело + Раздел 7» промоутился бы целиком. Аннотации переживают
    split/merge (срезы со пересчётом оффсетов), поэтому поздняя привязка
    попадает точнее — в конкретную строку якоря.
    """
    import logging

    from rag_parser.parsers.base import rows_to_markdown_table

    log = logging.getLogger(__name__)
    by_uid: Dict[str, DocumentNode] = {}
    early_orphans: List[DocumentNode] = []
    for table in tables or []:
        rows = table_to_rows(table)
        if not rows:
            continue
        md_table, _, _ = rows_to_markdown_table(rows)
        node = _table_node(md_table)
        uid = _table_uid(table)
        if uid and uid not in by_uid:
            by_uid[uid] = node
        else:
            early_orphans.append(node)
    rendered = set()

    def visit(node: DocumentNode) -> None:
        anchored: List[DocumentNode] = []
        for uid in _anchor_uids(
            node.metadata.get("annotations")
            if isinstance(node.metadata.get("annotations"), list)
            else []
        ):
            if uid in by_uid and uid not in rendered:
                anchored.append(by_uid[uid])
                rendered.add(uid)
        if anchored:
            node.children = [*anchored, *node.children]
        for child in node.children:
            visit(child)

    for child in children:
        visit(child)
    orphans: List[DocumentNode] = list(early_orphans)
    for uid, node in by_uid.items():
        if uid not in rendered:
            children.append(node)
            orphans.append(node)
    children.extend(early_orphans)
    if orphans:
        log.warning("orphan_tables", extra={"count": len(orphans)})
    return len(orphans)


def detach_tables(root: DocumentNode) -> None:
    """Убирает все узлы `type == "table"` из дерева (порядок остальных kept)."""

    def _drop(children: List[DocumentNode]) -> List[DocumentNode]:
        kept: List[DocumentNode] = []
        for child in children:
            if child.type == "table":
                continue
            child.children = _drop(child.children)
            kept.append(child)
        return kept

    root.children = _drop(root.children)


def reattach_tables(root: DocumentNode, tables: List[Any]) -> int:
    """Перепривязка таблиц после промоции + orphan-счётчик в metadata.

    Сносит старую привязку (поставленную до split/merge) и ставит заново
    по актуальным аннотациям строк. Возвращает число orphan-таблиц.
    """
    detach_tables(root)
    orphans = _attach_tables(root.children, tables)
    root.metadata["orphan_tables"] = orphans
    return orphans


def truncate(node: DocumentNode, preview_len: int = 30) -> DocumentNode:
    """Урезанная копия дерева для фронта: первые N символов + "…".

    Сырые аннотации (нужны только промоутеру) отбрасываются — остаётся
    только ``annotations_count``.
    """
    text = node.text or ""
    if len(text) > preview_len:
        text = text[:preview_len] + "…"
    metadata = {
        key: value for key, value in node.metadata.items() if key != "annotations"
    }
    return DocumentNode(
        type=node.type,
        level=node.level,
        text=text,
        children=[truncate(child, preview_len) for child in node.children],
        metadata=metadata,
    )


def build_preview_tree(
    structure: Any,
    tables: Optional[List[Any]] = None,
    preview_len: int = 30,
) -> DocumentNode:
    """Дерево с урезанными текстами (для API/UI). Чанкинг — по полному."""
    return truncate(build_full_tree(structure, tables), preview_len)


def iter_nodes(root: DocumentNode):
    """Обход дерева в глубину (сам узел + потомки)."""
    yield root
    for child in root.children:
        yield from iter_nodes(child)


def collect_headers(root: DocumentNode) -> List[DocumentNode]:
    """Все узлы ``type="header"`` в порядке обхода."""
    return [node for node in iter_nodes(root) if node.type == "header"]


def _markdown_for_node(node: DocumentNode, out: List[str]) -> None:
    import re

    text = (node.text or "").strip()
    if node.type == "header":
        level = max(1, min(6, node.level or 1))
        if text:
            out.append(f"{'#' * level} {text}")
    elif node.type == "table":
        if text:
            out.append(text)
    elif node.type == "list_item":
        if text:
            if re.match(r"^([-*+]\s|\d+[.)]\s)", text):
                out.append(text)
            else:
                out.append(f"- {text}")
    elif node.type in ("paragraph", "raw_text"):
        if text:
            out.append(text)
    for child in node.children:
        _markdown_for_node(child, out)


def full_tree_to_markdown(root: DocumentNode) -> str:
    """Markdown из ПОЛНОГО дерева (до урезания превью)."""
    import re

    parts: List[str] = []
    _markdown_for_node(root, parts)
    markdown = "\n\n".join(part for part in parts if part and part.strip())
    markdown = markdown.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n\n")
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip()
