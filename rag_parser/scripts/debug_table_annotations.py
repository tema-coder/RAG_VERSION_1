"""Диагностика TableAnnotation: какие форматы дают якоря таблиц, какие нет.

Печатает по каждому файлу: число таблиц, число аннотаций ``table`` в дереве,
сколько уникальных uid и сколько таблиц реально удалось привязать.

Запуск:
    python -m rag_parser.scripts.debug_table_annotations [файлы...]
    python -m rag_parser.scripts.debug_table_annotations            # весь input/
"""

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag_parser.parsers.dedoc_parser import SUPPORTED_EXTENSIONS, get_manager

DEFAULT_INPUT = Path(__file__).resolve().parent.parent.parent / "input"


def _anchors(node: Dict[str, Any], out: List[str], depth: int = 0) -> List[Dict[str, Any]]:
    """Собирает значения всех TableAnnotation в поддереве."""
    found: List[Dict[str, Any]] = []
    for annotation in node.get("annotations") or []:
        if annotation.get("name") == "table":
            out.append(annotation.get("value"))
            found.append(annotation)
    for child in node.get("subparagraphs") or []:
        found.extend(_anchors(child, out, depth + 1))
    return found


def analyse(path: Path) -> Dict[str, Any]:
    """Разбирает файл через dedoc и собирает статистику по якорям таблиц."""
    manager = get_manager()
    parsed = manager.parse(
        str(path), parameters={"with_attachments": "false", "structure_type": "tree"}
    )
    content = parsed.to_api_schema().model_dump().get("content") or {}
    tables = content.get("tables") or []
    structure = content.get("structure") or {}

    anchor_values: List[str] = []
    anchor_nodes = _anchors(structure, anchor_values)
    table_uids = [
        (table.get("metadata") or {}).get("uid") for table in tables
    ]
    unique_anchors = {value for value in anchor_values if value}
    unique_uids = {uid for uid in table_uids if uid}
    duplicates = len(anchor_values) - len(unique_anchors)

    # Сколько таблиц реально привяжется ПОСЛЕ нашего фикса очереди:
    # для каждого uid берётся столько таблиц, сколько встретилось аннотаций
    # (но не больше, чем самих таблиц). Здесь min(), а не множественное
    # пересечение — иначе для одинаковых таблиц показывало бы 1 вместо 9.
    per_uid_anchors: Dict[str, int] = {}
    for value in anchor_values:
        if value:
            per_uid_anchors[value] = per_uid_anchors.get(value, 0) + 1
    per_uid_tables: Dict[str, int] = {}
    for uid in table_uids:
        if uid:
            per_uid_tables[uid] = per_uid_tables.get(uid, 0) + 1
    attachable = sum(
        min(per_uid_anchors[uid], per_uid_tables.get(uid, 0))
        for uid in per_uid_anchors
    )
    collision_uids = sum(
        1 for uid, count in per_uid_tables.items() if count > 1
    )

    return {
        "file": path.name,
        "tables": len(tables),
        "annotations": len(anchor_values),
        "unique_anchors": len(unique_anchors),
        "unique_uids": len(unique_uids),
        "attachable": attachable,
        "orphan": len(tables) - attachable,
        "duplicate_anchors": duplicates,
        "collision_uids": collision_uids,
        "anchor_nodes": anchor_nodes,
    }


def default_files() -> List[Path]:
    if not DEFAULT_INPUT.is_dir():
        return []
    supported = {ext.lower() for ext in SUPPORTED_EXTENSIONS}
    return sorted(
        item
        for item in DEFAULT_INPUT.iterdir()
        if item.is_file() and item.suffix.lower() in supported
    )


def _fmt(value: Optional[str], width: int = 12) -> str:
    return (value or "-")[:width]


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    files = [Path(item) for item in args] if args else default_files()
    if not files:
        print("Нет файлов для анализа (передайте пути аргументами).")
        return 1

    header = (
        f"{'файл':26} {'табл':>5} {'аннот':>6} {'уник.':>6} {'привяз':>7} "
        f"{'orphan':>7} {'коллиз':>7}"
    )
    print(f"\n{'=' * len(header)}")
    print("  АНАЛИЗ TableAnnotation ПО ФОРМАТАМ")
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))

    for path in files:
        if not path.exists():
            print(f"{path.name:26} {'<файл не найден>':>5}")
            continue
        try:
            stats = analyse(path)
        except Exception as exc:  # noqa: BLE001 - диагностика не падает
            print(f"{path.name:26} {'ОШИБКА':>5}  {type(exc).__name__}: {exc}")
            continue
        print(
            f"{stats['file'][:26]:26} {stats['tables']:>5} {stats['annotations']:>6} "
            f"{stats['unique_anchors']:>6} {stats['attachable']:>7} "
            f"{stats['orphan']:>7} {stats['collision_uids']:>7}"
        )
        if stats["anchor_nodes"]:
            for node in stats["anchor_nodes"][:3]:
                print(
                    f"    якорь: node={node.get('node_id')} "
                    f"uid={_fmt(node.get('value'), 14)}"
                )

    print("\nВыводы:")
    print("  аннот == 0 -> dedoc НЕ даёт якорей для этого формата; таблица")
    print("               физически не привязывается и уходит в orphan")
    print("  коллиз > 0 -> несколько ОДИНАКОВЫХ таблиц имеют одинаковый uid")
    print("               (баг dedoc: md5(xml) без дедупликации). Лечится")
    print("               очередью аннотаций в _attach_tables")
    print("  привяз     -> сколько таблиц реально привяжется ПОСЛЕ фикса")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI-точка входа
    raise SystemExit(main())


__all__ = ["analyse", "default_files", "main"]