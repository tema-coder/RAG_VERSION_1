"""Диагностика привязки таблиц в HTML.

Показывает, что происходит с таблицами на маршруте
``HtmlReader + маркеры __RAGTABLE_N__``:

* сколько таблиц было в исходном HTML и сколько извлечено;
* какие заголовки распознаны и сколько таблиц висит под каждым;
* какие таблицы остались orphan и почему.

Запуск:
    python -m rag_parser.scripts.debug_html_tables [файл.html ...]
"""

import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bs4 import BeautifulSoup

from rag_parser.parsers.html_preprocessor import inject_table_markers

logger = logging.getLogger(__name__)


def _count_source_tables(path: Path) -> int:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    return len(soup.find_all("table"))


def _report(path: Path) -> int:
    from rag_parser.parsers.dedoc_parser import DedocParser

    source_tables = _count_source_tables(path)
    _, extracted = inject_table_markers(
        path.read_text(encoding="utf-8", errors="replace")
    )

    result = DedocParser().parse(path)
    tree = result.tree

    attached = int(tree.metadata.get("tables_by_marker", 0) or 0)
    orphans = int(tree.metadata.get("orphan_tables", 0) or 0)

    print(f"\n{'=' * 72}")
    print(f"  {path.name}")
    print(f"{'=' * 72}")
    print(f"таблиц в исходном HTML : {source_tables}")
    print(f"извлечено маршрутом     : {len(extracted)}")
    print(f"привязано по маркерам   : {attached}")
    print(f"orphan                  : {orphans}")
    if orphans:
        print("  ВНИМАНИЕ: таблица выше первого заголовка — привязать не к чему")

    print("\nдерево (заголовки и таблицы):")
    header = None
    found = 0

    def walk(node, depth=0, parent_is_root=False):
        """Печатает дерево; ``parent_is_root`` отличает orphan-таблицы."""
        nonlocal header, found
        if node.type == "header" and node.text.strip():
            header = node.text.strip()
            print(f"{'  ' * depth}# {header[:70]}")
        if node.type == "table":
            found += 1
            head = node.text.splitlines()[0][:60] if node.text else ""
            owner = "ORPHAN (вне разделов)" if parent_is_root else repr(header)
            print(f"{'  ' * depth}[TABLE] под {owner}")
            print(f"{'  ' * depth}         {head}")
        for child in node.children:
            walk(child, depth + 1, parent_is_root=(node is tree))

    walk(tree)
    if found == 0 and source_tables:
        print("  (таблицы не найдены в дереве — проверьте маркеры)")
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    logging.disable(logging.CRITICAL)
    args = list(argv if argv is not None else sys.argv[1:])
    files = [Path(item) for item in args] or [
        item
        for item in sorted(
            (Path(__file__).resolve().parent.parent.parent / "input").glob("*.html*")
        )
    ]
    files: List[Path] = [item for item in files if item.exists()]
    if not files:
        print("HTML-файлы не найдены. Передайте путь аргументом.")
        return 1
    for path in files:
        _report(path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI-точка входа
    raise SystemExit(main())


__all__ = ["main"]