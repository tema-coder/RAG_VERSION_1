"""Диагностика дублирования/потери таблиц в разделах.

Печатает по каждому узлу таблицы: родительский заголовок, индекс в
``children`` родителя и позицию в обходе документа. Отдельно — сколько
таблиц ожидается, сколько осталось orphan и не осталось ли где-нибудь
неудалённых маркеров ``__RAGTABLE_N__``.

Запуск:
    python -m rag_parser.scripts.debug_table_duplication
    python -m rag_parser.scripts.debug_table_duplication input/gost_report.md
"""

import logging
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag_parser.core.interfaces import DocumentNode
from rag_parser.parsers.html_preprocessor import MARKER_PREFIX
from rag_parser.parsers.dedoc_parser import DedocParser

logger = logging.getLogger(__name__)

#: Раздел -> сколько таблиц в нём должно быть (эталон gost_report).
EXPECTED_BY_SECTION = {
    "1": 1,
    "5": 1,
    "5.1": 1,
    "5.2": 1,
    "6.1": 1,
    "6.2.1": 1,
    "6.2.2": 1,
    "6.2.3": 1,
    "7": 1,
}

INPUT = Path(__file__).resolve().parent.parent.parent / "input"


def _section_number(text: str) -> Optional[str]:
    match = re.match(r"^\s*(\d+(?:\.\d+)*)", text or "")
    return match.group(1) if match else None


def collect_tables(root: DocumentNode) -> Tuple[List[dict], int, List[str]]:
    """Собирает таблицы, счётчики и тексты с маркерами.

    :return: ``(таблицы, всего_таблиц, абзацы_с_маркерами)``
    """
    tables: List[dict] = []
    leaked: List[str] = []
    counter = 0

    def walk(node: DocumentNode, parent: Optional[DocumentNode], index: int) -> None:
        nonlocal counter
        if node.type == "table":
            counter += 1
            tables.append(
                {
                    "n": counter,
                    "parent": parent.text.strip() if parent else "(вне разделов)",
                    "index": index,
                    "header": (node.text.splitlines() or [""])[0][:44],
                }
            )
        if MARKER_PREFIX in (node.text or ""):
            leaked.append(node.text.strip()[:60])
        for position, child in enumerate(node.children):
            walk(child, node if node.type == "header" else parent, position)

    walk(root, None, 0)
    return tables, counter, leaked


def _report(path: Path) -> bool:
    tree = DedocParser().parse(path).tree
    tables, total, leaked = collect_tables(tree)
    orphan = int(tree.metadata.get("orphan_tables", 0) or 0)

    print(f"\n{'=' * 74}")
    print(f"  {path.name}")
    print(f"{'=' * 74}")
    print(f"таблиц в дереве : {total}")
    print(f"orphan           : {orphan}")
    print(f"маркеры в тексте : {len(leaked)}" + (f" -> {leaked[:3]}" if leaked else ""))

    print(f"\n{'#':>3}  {'индекс':>6}  {'раздел':<40} первая строка")
    for item in tables:
        print(f"{item['n']:>3}  {item['index']:>6}  {item['parent'][:40]:<40} {item['header']}")

    # Сверка с эталоном: сколько таблиц в каждом разделе.
    counts = {}
    for item in tables:
        number = _section_number(item["parent"])
        if number:
            counts[number] = counts.get(number, 0) + 1

    mismatches = []
    for number, expected in EXPECTED_BY_SECTION.items():
        actual = counts.get(number, 0)
        if actual != expected:
            mismatches.append(f"раздел {number}: ожидалось {expected}, получено {actual}")

    if mismatches:
        print("\nПРОБЛЕМЫ с привязкой:")
        for line in mismatches:
            print(f"  - {line}")
        return False

    print("\nВЫВОД: каждая таблица в своём разделе, дублирования нет.")
    return True


def main(argv: Optional[List[str]] = None) -> int:
    logging.disable(logging.CRITICAL)
    args = list(argv if argv is not None else sys.argv[1:])
    files = [Path(item) for item in args] or [
        INPUT / name
        for name in ("gost_report.md", "gost_report.html", "gost_report.txt",
                     "gost_report.xml", "gost_report.docx")
    ]
    files = [item for item in files if item.exists()]
    if not files:
        print("Файлы не найдены. Передайте путь аргументом.")
        return 1
    return 0 if all(_report(path) for path in files) else 1


if __name__ == "__main__":  # pragma: no cover - CLI-точка входа
    raise SystemExit(main())


__all__ = ["collect_tables", "main"]