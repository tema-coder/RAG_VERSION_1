"""Эвристика промоции «визуальный заголовок → header».

Реальные docx часто оформлены без стилей Word: заголовки выделены
визуально (bold, размер, центр, нумерация «Раздел N.»), а dedoc отдаёт их
как ``raw_text``/``paragraph`` — иногда даже склеенными с телом в один
многострочный узел (аннотации при этом привязаны к диапазонам символов).
Модуль чинит это до генерации Markdown и чанкинга (проход 0 — сканы,
затем рекурсия, затем глобальный ребилд):

0. все листья режутся на строки; по порядку строк считаются distinct
   «Раздел N» (супрессия `N)` при ≥2) и TOC-серии (≥5 подряд `Раздел N`
   с N строго +1 — не промоутятся);
1. каждая однострочная нода скорингуется;
2. ноды с баллом ``>= threshold`` получают ``type="header"``;
3. соседние непромоутнутые строки склеиваются обратно в абзацы
   (TOC-строки остаются поштучно);
4. дерево сплющивается в порядок документа и пересобирается стеком
   по уровням (lift промоутнутых заголовков из чужой вложенности).

Шкала (из ТЗ):

| Признак | Баллы |
|---|---|
| Regex нумерации ``^(Раздел\\s+\\d+\\.?\|\\d+(\\.\\d+)*\\.?)(\\s\|$)`` / ``^\\d+\\)`` | +4 явный «Раздел N», иначе +3; level из `numbering_level` |
| Аннотация ``bold=True`` | +2 |
| ``size > 14`` | +1.5 |
| Текст в КАПСЕ | +1 |
| Длина < 100 символов | +1 |
| ``alignment=center`` | +0.5 |
| Заканчивается точкой и длиннее 120 символов | −2 |

Уровень из нумерации: точки считаются только между цифрами
(«Раздел 2.» → 1, «2.» → 1, «2.3» → 2, «2.3.1» → 3); ветка ``N)`` → 2.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from rag_parser.core.interfaces import DocumentNode

NUMBERING_RE = re.compile(
    r"^(Раздел\s+\d+\.?|\d+(\.\d+)*\.?)(\s|$)|^\d+\)", re.IGNORECASE
)
RAZDEL_N_RE = re.compile(r"^Раздел\s+(\d+)", re.IGNORECASE)

TOC_SERIES_MIN = 5


def _annotations_of(node: DocumentNode) -> List[Dict[str, Any]]:
    raw = node.metadata.get("annotations")
    if not isinstance(raw, list):
        return []
    result: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            result.append(item)
            continue
        result.append(
            {
                "name": getattr(item, "name", ""),
                "start": getattr(item, "start", 0),
                "end": getattr(item, "end", 0),
                "value": getattr(item, "value", ""),
            }
        )
    return result


def _coverage(annotations: List[Dict[str, Any]], name: str, length: int) -> Tuple[bool, Any]:
    """Есть ли аннотация ``name``, покрывающая >=60% строки. Возвращает (да, value)."""
    if length <= 0:
        return False, None
    best = 0.0
    best_value: Any = None
    for item in annotations:
        if str(item.get("name", "")).lower() != name:
            continue
        try:
            start = int(item.get("start", 0) or 0)
            end = int(item.get("end", 0) or 0)
        except (TypeError, ValueError):
            continue
        overlap = max(0, min(end, length) - max(start, 0))
        ratio = overlap / length
        if ratio > best:
            best = ratio
            best_value = item.get("value")
    return best >= 0.6, best_value


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "bold")


def numbering_level(matched: re.Match) -> int:
    """Уровень заголовка из regex-нумерации.

    Точки считаются только между группами цифр: «Раздел 2.» → 1,
    «2.» → 1, «2.3» → 2, «2.3.1» → 3. Ветка ``N)`` (group(1) is None) —
    подчинённый уровень 2.
    """
    group = matched.group(1)
    if group is None:
        return 2
    number = re.sub(r"^Раздел\s+", "", group, flags=re.IGNORECASE).strip()
    number = number.rstrip(".").rstrip(")")
    return 1 + len(re.findall(r"(?<=\d)\.(?=\d)", number))


def score_node(
    text: str,
    annotations: List[Dict[str, Any]],
    allow_paren: bool = True,
) -> Tuple[float, Optional[int]]:
    """Балл узла + уровень из нумерации (если есть).

    При ``allow_paren=False`` ветка ``N)`` не даёт ни баллов, ни уровня
    (двухпроходная супрессия списков в «Раздел N»-документах).
    """
    stripped = (text or "").strip()
    score = 0.0
    level: Optional[int] = None
    if not stripped:
        return 0.0, None

    match = NUMBERING_RE.match(stripped)
    if match and (allow_paren or match.group(1) is not None):
        if match.group(1) is not None and RAZDEL_N_RE.match(
            match.group(1).strip()
        ):
            # Явный маркер раздела — промоутится независимо от длины:
            # длинные титулы («Раздел 5. Отчет ... по приоритетным ...»)
            # иначе не добирают балл и их секции (включая таблицы)
            # приписываются предыдущему разделу.
            score += 4.0
        else:
            score += 3.0
        level = numbering_level(match)

    length = len(stripped)
    bold, _ = _coverage(annotations, "bold", length)
    if bold:
        # bold считается только при truthy-значении хотя бы у одной покрывающей аннотации
        for item in annotations:
            if str(item.get("name", "")).lower() != "bold":
                continue
            try:
                start = int(item.get("start", 0) or 0)
                end = int(item.get("end", 0) or 0)
            except (TypeError, ValueError):
                continue
            if length and (min(end, length) - max(start, 0)) / length >= 0.6:
                if _is_truthy(item.get("value")):
                    score += 2.0
                    break

    size_hit, size_value = _coverage(annotations, "size", length)
    if size_hit:
        try:
            if float(str(size_value).replace(",", ".")) > 14:
                score += 1.5
        except (TypeError, ValueError):
            pass

    letters = [ch for ch in stripped if ch.isalpha()]
    if len(letters) >= 4 and all(ch.isupper() for ch in letters):
        score += 1.0

    if length < 100:
        score += 1.0

    center_hit, center_value = _coverage(annotations, "alignment", length)
    if center_hit and str(center_value).strip().lower() == "center":
        score += 0.5

    if length > 120 and stripped.endswith("."):
        score -= 2.0

    return score, level


def _slice_annotations(
    annotations: List[Dict[str, Any]], start: int, end: int
) -> List[Dict[str, Any]]:
    sliced: List[Dict[str, Any]] = []
    for item in annotations:
        try:
            item_start = int(item.get("start", 0) or 0)
            item_end = int(item.get("end", 0) or 0)
        except (TypeError, ValueError):
            continue
        if item_end <= start or item_start >= end:
            continue
        piece = dict(item)
        piece["start"] = max(item_start, start) - start
        piece["end"] = min(item_end, end) - start
        sliced.append(piece)
    return sliced


def _split_lines(node: DocumentNode) -> List[DocumentNode]:
    """Режет многострочный узел на однострочные (аннотации пересчитываются)."""
    if node.type in ("table", "document") or "\n" not in (node.text or ""):
        return [node]
    annotations = _annotations_of(node)
    if not annotations:
        # Без аннотаций делить не по чему — делим вслепую.
        return [
            DocumentNode(
                type=node.type,
                level=node.level,
                text=line.strip(),
                children=[],
                metadata={"annotations_count": 0, "annotations": []},
            )
            for line in node.text.split("\n")
            if line.strip()
        ]
    pieces: List[DocumentNode] = []
    offset = 0
    for line in node.text.split("\n"):
        line_len = len(line)
        sliced = _slice_annotations(annotations, offset, offset + line_len)
        if line.strip():
            pieces.append(
                DocumentNode(
                    type=node.type,
                    level=node.level,
                    text=line.strip(),
                    children=[],
                    metadata={
                        "annotations_count": len(sliced),
                        "annotations": sliced,
                    },
                )
            )
        offset += line_len + 1  # +1 за "\n"
    return pieces or [node]


def _merge_paragraphs(nodes: List[DocumentNode]) -> List[DocumentNode]:
    """Склеивает соседние непромоутнутые строки обратно в абзацы."""
    merged: List[DocumentNode] = []
    buffer: List[DocumentNode] = []

    def flush() -> None:
        if not buffer:
            return
        if len(buffer) == 1:
            merged.append(buffer[0])
        else:
            text = "\n".join(item.text for item in buffer)
            annotations: List[Dict[str, Any]] = []
            offset = 0
            for item in buffer:
                for ann in _annotations_of(item):
                    piece = dict(ann)
                    try:
                        piece["start"] = int(piece.get("start", 0) or 0) + offset
                        piece["end"] = int(piece.get("end", 0) or 0) + offset
                    except (TypeError, ValueError):
                        continue
                    annotations.append(piece)
                offset += len(item.text) + 1
            first = buffer[0]
            merged.append(
                DocumentNode(
                    type="paragraph" if first.type != "raw_text" else "raw_text",
                    level=first.level,
                    text=text,
                    children=[],
                    metadata={
                        "annotations_count": len(annotations),
                        "annotations": annotations,
                    },
                )
            )
        buffer.clear()

    for item in nodes:
        if (
            item.type == "header"
            or item.children
            or item.type == "table"
            or item.metadata.get("toc")
        ):
            flush()
            merged.append(item)
        else:
            buffer.append(item)
    flush()
    return merged


def _flatten(root: DocumentNode) -> List[DocumentNode]:
    """Дерево в плоский список узлов в порядке документа (pre-order)."""
    out: List[DocumentNode] = []

    def visit(node: DocumentNode) -> None:
        out.append(node)
        for child in node.children:
            visit(child)

    for child in root.children:
        visit(child)
    return out


def _rebuild(nodes: List[DocumentNode]) -> List[DocumentNode]:
    """Стековая сборка детей из плоского списка по уровням заголовков.

    Промоутнутый заголовок всплывает из чужой вложенности (lift): его место
    определяется только уровнем, а не dedoc-родителем. В стек пушатся только
    ``type == "header"`` с ``level >= 1`` — header с level 0 не становится
    родителем последующего (не поглощает абзацы).
    """
    result: List[DocumentNode] = []
    stack: List[DocumentNode] = []
    for source in nodes:
        node = DocumentNode(
            type=source.type,
            level=source.level,
            text=source.text,
            children=[],
            metadata=dict(source.metadata),
        )
        if node.type == "header" and (node.level or 0) >= 1:
            while stack and stack[-1].level >= node.level:
                stack.pop()
            if stack:
                stack[-1].children.append(node)
            else:
                result.append(node)
            stack.append(node)
        else:
            if stack:
                stack[-1].children.append(node)
            else:
                result.append(node)
    return result


def _presplit(node: DocumentNode) -> List[DocumentNode]:
    """Заменяет многострочные листья однострочными сиблингами (порядок)."""
    if node.type in ("table", "document"):
        return [
            DocumentNode(
                type=node.type,
                level=node.level,
                text=node.text,
                children=[
                    piece
                    for child in node.children
                    for piece in _presplit(child)
                ],
                metadata=dict(node.metadata),
            )
        ]
    if node.children:
        return [
            DocumentNode(
                type=node.type,
                level=node.level,
                text=node.text,
                children=[
                    piece
                    for child in node.children
                    for piece in _presplit(child)
                ],
                metadata=dict(node.metadata),
            )
        ]
    return _split_lines(node)


def _is_suppressed_paren(text: str, allow_paren: bool) -> bool:
    """Строка `N)` при запрете paren-ветки (двухпроходная супрессия)."""
    if allow_paren:
        return False
    match = NUMBERING_RE.match((text or "").strip())
    return match is not None and match.group(1) is None


def _count_razdel(lines: List[DocumentNode]) -> int:
    """Число distinct «Раздел N» среди однострочных листьев."""
    numbers = set()
    for line in lines:
        match = RAZDEL_N_RE.match((line.text or "").strip())
        if match:
            numbers.add(match.group(1))
    return len(numbers)


def _mark_toc(lines: List[DocumentNode]) -> int:
    """Помечает TOC-серии (≥5 подряд «Раздел N» с N строго +1). Возвращает число."""
    count = 0
    run: List[DocumentNode] = []
    last_number = 0

    def flush_run() -> None:
        nonlocal count
        if len(run) >= TOC_SERIES_MIN:
            for item in run:
                item.metadata["toc"] = True
                count += 1

    for line in lines:
        match = RAZDEL_N_RE.match((line.text or "").strip())
        if match:
            number = int(match.group(1))
            if run and number == last_number + 1:
                run.append(line)
            else:
                flush_run()
                run = [line]
            last_number = number
        else:
            flush_run()
            run = []
            last_number = 0
    flush_run()
    return count


def promote_headers(root: DocumentNode, threshold: Optional[float] = None) -> DocumentNode:
    """Промоут визуальных заголовков. Не мутирует вход (возвращает новое дерево)."""
    from rag_parser.core.config import settings

    limit = float(settings.HEADER_PROMOTION_THRESHOLD if threshold is None else threshold)

    # Проход 0: пре-сплит всех листьев + сканы по порядку строк.
    split_children: List[DocumentNode] = []
    for child in root.children:
        split_children.extend(_presplit(child))
    scan: List[DocumentNode] = []

    def _collect(nodes: List[DocumentNode]) -> None:
        for item in nodes:
            if item.type == "document":
                _collect(item.children)
            else:
                scan.append(item)
                _collect(item.children)

    _collect(split_children)
    scan_lines = [node for node in scan if not node.children]
    allow_paren = _count_razdel(scan_lines) < 2
    toc_suppressed = _mark_toc(scan_lines)
    paren_suppressed = 0

    def process_children(children: List[DocumentNode]) -> List[DocumentNode]:
        """Рекурсивно обрабатывает список детей, возвращает новый список."""
        out: List[DocumentNode] = []
        for child in children:
            out.extend(process_one(child))
        return out

    def process_one(node: DocumentNode) -> List[DocumentNode]:
        nonlocal paren_suppressed
        if node.type in ("table", "document"):
            return [
                DocumentNode(
                    type=node.type,
                    level=node.level,
                    text=node.text,
                    children=process_children(node.children),
                    metadata=dict(node.metadata),
                )
            ]
        if node.children:
            sub = process_children(node.children)
            if _is_suppressed_paren(node.text, allow_paren):
                paren_suppressed += 1
                node_type, level = node.type, node.level
            else:
                score, numbered_level = score_node(
                    node.text, _annotations_of(node), allow_paren
                )
                node_type, level = node.type, node.level
                if (
                    node_type != "header"
                    and node.text.strip()
                    and not node.metadata.get("toc")
                    and score >= limit
                ):
                    node_type, level = "header", numbered_level or 1
            return [
                DocumentNode(
                    type=node_type,
                    level=level,
                    text=node.text,
                    children=sub,
                    metadata=dict(node.metadata),
                )
            ]
        # лист (после пре-сплита — однострочный): скорим → клеим абзацы
        lines = _split_lines(node)
        scored: List[DocumentNode] = []
        for line in lines:
            if line.metadata.get("toc"):
                scored.append(line)
                continue
            if _is_suppressed_paren(line.text, allow_paren):
                paren_suppressed += 1
                scored.append(line)
                continue
            score, numbered_level = score_node(
                line.text, _annotations_of(line), allow_paren
            )
            if line.type != "header" and score >= limit:
                scored.append(
                    DocumentNode(
                        type="header",
                        level=numbered_level or 1,
                        text=line.text,
                        children=[],
                        metadata=dict(line.metadata),
                    )
                )
            else:
                scored.append(line)
        return _merge_paragraphs(scored)

    processed = DocumentNode(
        type="document",
        level=0,
        text=root.text,
        children=process_children(split_children),
        metadata=dict(root.metadata),
    )
    metadata = dict(processed.metadata)
    metadata["toc_suppressed"] = toc_suppressed
    metadata["paren_suppressed"] = paren_suppressed
    return DocumentNode(
        type="document",
        level=0,
        text=processed.text,
        children=_rebuild(_flatten(processed)),
        metadata=metadata,
    )


__all__ = ["NUMBERING_RE", "promote_headers", "score_node"]
