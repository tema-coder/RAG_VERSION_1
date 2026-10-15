"""Диагностика: что происходит с таблицами при конвертации PDF → DOCX.

Задача 6 ТЗ: сравнить порядок узлов для исходного DOCX и для DOCX,
полученного из PDF. Нужно понять, чья это проблема — конвертера или dedoc.

Что именно сравнивается
-----------------------
Сырой JSON dedoc для DOCX **не содержит** таблиц в дереве: они лежат
отдельным списком ``content.tables`` и в ``structure`` их просто нет.
Поэтому «порядок узлов в сыром JSON» для таблиц не определён, и сравнивать
нечего. Сравниваем то, что реально видит пользователь: **последовательность
секций, к которым привязаны таблицы**, в итоговом дереве пайплайна
(``DedocParser``), плюс сырые счётчики (сколько текстовых блоков и сколько
таблиц вернул dedoc).

Запуск:
    python -m rag_parser.scripts.debug_pdf_via_docx input/gost_report.pdf
    python -m rag_parser.scripts.debug_pdf_via_docx input/gost_report.pdf --original input/gost_report.docx
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag_parser.parsers.dedoc_parser import DedocParser

logger = logging.getLogger(__name__)

_NUMBERING = re.compile(r"^\s*\d+(?:\.\d+)*")


def _raw_summary(path: Path) -> Dict[str, int]:
    """Сырые счётчики dedoc: текстовые блоки и таблицы."""
    from rag_parser.parsers.dedoc_parser import parameters_for_format, prepare_manager_for_format

    manager = prepare_manager_for_format(".docx")
    parsed = manager.parse(str(path), parameters=parameters_for_format(".docx", path))
    content = (parsed.to_api_schema().model_dump().get("content") or {})
    structure = content.get("structure") or {}
    texts: List[str] = []

    def walk(node: Any) -> None:
        """В dedoc структура хранится в ``subparagraphs`` (не ``children``)."""
        if not isinstance(node, dict):
            return
        text = node.get("text")
        if text:
            texts.append(text)
        for child in node.get("subparagraphs") or []:
            walk(child)

    walk(structure)
    return {
        "text_blocks": len(texts),
        "tables": len(content.get("tables") or []),
    }


def _is_header(text: str) -> bool:
    """Заголовок по тем же признакам, что и ``header_promoter``."""
    stripped = (text or "").strip()
    if not stripped or len(stripped) > 120:
        return False
    if _NUMBERING.match(stripped):
        return True
    letters = [ch for ch in stripped if ch.isalpha()]
    return bool(letters) and all(ch.isupper() for ch in letters)


def _anchor_sequence(path: Path) -> Tuple[List[int], List[str]]:
    """Последовательность заголовков, под которыми лежат таблицы."""
    from rag_parser.core.interfaces import DocumentNode

    tree = DedocParser().parse(path).tree
    sections: List[str] = []
    anchors: List[int] = []
    current = -1
    fallback: List[str] = []

    def walk(node: DocumentNode) -> None:
        nonlocal current
        if node.type == "header" and node.text.strip():
            sections.append(node.text.strip())
            current = len(sections) - 1
        elif node.type == "table":
            if current >= 0:
                anchors.append(current)
            else:
                # таблица выше первого заголовка — привязать не к чему
                anchors.append(-1)
                fallback.append(node.text.splitlines()[0][:40] if node.text else "")
        for child in node.children:
            walk(child)

    walk(tree)
    if fallback:
        logger.info("orphan_tables", extra={"count": len(fallback)})
    return anchors, sections


def _describe(sections: List[str], index: int) -> str:
    if index < 0:
        return "ORPHAN (выше первого заголовка)"
    if index >= len(sections):
        return f"раздел #{index} (не найден)"
    return sections[index][:45]


def _report(original: Path, converted: Path) -> int:
    left_raw = _raw_summary(original)
    right_raw = _raw_summary(converted)
    left, left_sections = _anchor_sequence(original)
    right, right_sections = _anchor_sequence(converted)

    print(f"\n{'=' * 74}")
    print(f"  {original.name}  vs  {converted.name}")
    print(f"{'=' * 74}")
    print(f"{'':22} {'исходный DOCX':>16} {'DOCX из PDF':>14}")
    print(f"{'текстовых блоков':22} {left_raw['text_blocks']:>16} {right_raw['text_blocks']:>14}")
    print(f"{'таблиц (сырой dedoc)':22} {left_raw['tables']:>16} {right_raw['tables']:>14}")
    print(f"{'таблиц в дереве':22} {len(left):>16} {len(right):>14}")
    print(f"{'orphan-таблиц':22} {left.count(-1):>16} {right.count(-1):>14}")

    print("\nтаблицы по разделам:")
    for index in range(max(len(left), len(right))):
        a = _describe(left_sections, left[index]) if index < len(left) else "—"
        b = _describe(right_sections, right[index]) if index < len(right) else "—"
        flag = " " if a == b else "  <-- РАСХОЖДЕНИЕ"
        print(f"  таблица {index}: {a:<47} | {b}{flag}")

    if left == right:
        print("\nВЫВОД: порядок таблиц СОХРАНЁН.")
        return 0

    print("\nВЫВОД: порядок таблиц НАРУШЕН.")
    if len(left) != len(right):
        print(
            f"  число таблиц разошлось: {len(left)} против {len(right)} — "
            "часть таблиц потеряна или раздвоена конвертером"
        )
    return 1


def _convert(pdf_path: Path, out_path: Path) -> Optional[Path]:
    try:
        import pdf2docx
    except ImportError:
        print("pdf2docx не установлен — конвертация невозможна.", file=sys.stderr)
        return None
    converter = pdf2docx.Converter(str(pdf_path))
    try:
        converter.convert(str(out_path))
    finally:
        converter.close()
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    logging.disable(logging.CRITICAL)
    parser = argparse.ArgumentParser(
        description="Диагностика PDF -> DOCX и порядка таблиц"
    )
    parser.add_argument("pdf", type=Path, help="исходный PDF")
    parser.add_argument("--original", type=Path, default=None, help="DOCX для сравнения")
    parser.add_argument("--keep", action="store_true", help="не удалять промежуточный DOCX")
    args = parser.parse_args(argv)

    if not args.pdf.exists():
        print(f"Файл не найден: {args.pdf}", file=sys.stderr)
        return 1

    original = args.original
    if original is None:
        candidate = args.pdf.with_suffix(".docx")
        original = candidate if candidate.exists() else None
    if original is None:
        print(
            "Не с чем сравнивать: не найден DOCX с тем же именем. "
            "Передайте --original.",
            file=sys.stderr,
        )
        return 1

    converted = Path("/tmp") / f"{args.pdf.stem}_via_pdf2docx.docx"
    if _convert(args.pdf, converted) is None:
        return 1

    try:
        return _report(original, converted)
    finally:
        if not args.keep:
            converted.unlink(missing_ok=True)


if __name__ == "__main__":  # pragma: no cover - CLI-точка входа
    raise SystemExit(main())