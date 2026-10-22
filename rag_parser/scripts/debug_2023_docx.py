"""Диагностика разбора 2023.docx (Шаг 1–2 плана диагностики task.txt).

Сквозной прогон ``DedocParser → promote_headers → HybridRagChunker`` с
печатью распознанных заголовков (уровень, глубина, toc-флаг) и чанков
(breadcrumb, parent_id, превью текста). Отвечает на вопросы диагностики:

* сколько всего заголовков распознано;
* сколько из них помечены как TOC;
* распознан ли «Раздел N» как заголовок, какой у него level и toc-статус;
* какой parent_id / breadcrumb у чанков «2.1. Полное наименование…».

Запуск::

    python -m rag_parser.scripts.debug_2023_docx path/to/2023.docx
    python -m rag_parser.scripts.debug_2023_docx            # по умолчанию input/2023.docx

Не мутирует исходники; читает готовые модули парсера.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from rag_parser.core.interfaces import DocumentNode


def _collect_headers(root: DocumentNode) -> List[dict]:
    """Все узлы ``type="header"`` в порядке обхода (pre-order)."""
    out: List[dict] = []

    def visit(node: DocumentNode, depth: int) -> None:
        if node.type == "header":
            out.append(
                {
                    "depth": depth,
                    "level": node.level,
                    "text": (node.text or "").strip(),
                    "toc": bool(node.metadata.get("toc")),
                }
            )
        for child in node.children:
            visit(child, depth + 1)

    for child in root.children:
        visit(child, 1)
    return out


def _print_headers(root: DocumentNode) -> None:
    headers = _collect_headers(root)
    print("=== ЗАГОЛОВКИ ===")
    for h in headers:
        toc = " [TOC]" if h["toc"] else ""
        print(f"  L{h['level']} d{h['depth']}: {h['text'][:70]}{toc}")
    print(f"\nВсего заголовков: {len(headers)}")
    print(f"TOC заголовков:   {sum(1 for h in headers if h['toc'])}")

    # Целевой диагноз: что стало с «Раздел N» в теле документа.
    print("\n=== «Раздел N» В РЕЗУЛЬТАТЕ ===")
    for h in headers:
        t = h["text"]
        if t.lower().startswith("раздел ") and any(ch.isdigit() for ch in t[:14]):
            print(f"  L{h['level']} d{h['depth']} toc={h['toc']}: {t[:75]}")


def _print_chunks(path: Path, limit: int = 30) -> None:
    """Чанки с breadcrumb/parent_id (через HybridRagChunker, tree-путь)."""
    from rag_parser.chunkers.hybrid import HybridRagChunker
    from rag_parser.parsers.dedoc_parser import DedocParser

    parser = DedocParser()
    result = parser.parse(path)
    chunker = HybridRagChunker(text_strategy="token")
    chunks = chunker.chunk(
        result.markdown,
        {"source": path.name, "format": result.format},
        tree=result.tree,
    )
    print(f"\n=== ЧАНКИ (всего {len(chunks)}, показано первых {limit}) ===")
    for i, chunk in enumerate(chunks[:limit]):
        print(f"Chunk #{i}: breadcrumb={chunk.breadcrumb!r} parent={chunk.parent_id}")
        print(f"  text={chunk.text[:70]!r}")

    # Прицельный поиск чанка с «2.1. Полное наименование» — индикатор бага.
    print("\n=== ЧАНКИ С «2.1» / «Полное наименование» ===")
    for i, chunk in enumerate(chunks):
        if "2.1" in chunk.text[:6] or "Полное наименование" in chunk.text:
            print(
                f"Chunk #{i}: breadcrumb={chunk.breadcrumb!r} parent={chunk.parent_id}"
            )
            print(f"  text={chunk.text[:90]!r}")
            break


def debug_2023_docx(path: Path) -> None:
    from rag_parser.parsers.dedoc_parser import DedocParser

    result = DedocParser().parse(path)
    print(f"=== {path.name}: parsed, markdown={len(result.markdown)} chars ===")
    if result.tree is None:
        print("NO TREE")
        return
    print(f"tree: {len(result.tree.children)} top-level children")
    print(
        f"toc_suppressed={result.tree.metadata.get('toc_suppressed')} "
        f"paren_suppressed={result.tree.metadata.get('paren_suppressed')}"
    )
    _print_headers(result.tree)
    _print_chunks(path)


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    path = Path(argv[0]) if argv else Path("input/2023.docx")
    if not path.exists():
        fallback = Path("/home/z/my-project/upload/2023.docx")
        if fallback.exists():
            path = fallback
        else:
            print(f"File not found: {path}", file=sys.stderr)
            return 2
    debug_2023_docx(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
