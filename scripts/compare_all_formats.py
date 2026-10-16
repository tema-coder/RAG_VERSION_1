#!/usr/bin/env python3
"""Сравнение всех форматов из ``input/`` с эталоном ``gost_report.docx``.

Эталон разобран и разбит на чанки один раз; его артефакты лежат в
``input/etalon/`` (``gost_report.tree.json``, ``gost_report.chunks.json``,
``gost_report.md``) — в отдельной папке, чтобы ``gost_report.md`` эталона не
оказался на месте входного файла ``input/gost_report.md``. Каждый формат из
``input/`` прогоняется через тот же
пайплайн ``DedocParser`` -> ``HybridRagChunker`` и сравнивается с эталоном по:

* количеству и уровням заголовков (с выравниванием, а не по позиции);
* количеству таблиц и orphan-таблицам;
* счётчику подавленных строк оглавления;
* количеству чанков, их breadcrumb и тексту.

Запуск (из корня репозитория):

    rag_parser/.venv/bin/python scripts/compare_all_formats.py
    rag_parser/.venv/bin/python scripts/compare_all_formats.py --format html
    rag_parser/.venv/bin/python scripts/compare_all_formats.py --strict

Результат печатается таблицей и сохраняется в
``output/comparison_results.json``.

Про допуски
-----------
Разные форматы физически не могут дать байт-в-байт тот же текст, что и
эталонный DOCX: в ``.md``/``.xml`` нет титульных строк «ПО ТЕХНИЧЕСКОМУ
РЕГУЛИРОВАНИЮ И МЕТРОЛОГИИ» и «НАЦИОНАЛЬНЫЙ СТАНДАРТ РОССИЙСКОЙ ФЕДЕРАЦИИ», в
``.txt`` весь текст в верхнем регистре, в ``.xml`` три строки заголовка лежат
в одном ``<title>``, а pdf2docx склеивает слова. Такие расхождения перечислены
в :data:`KNOWN_SOURCE_DELTAS` и показываются отдельным блоком ``ДОПУЩЕНО`` —
они не маскируются внутри сравнения и видны в отчёте. Флаг ``--strict``
считает их ошибками.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # запуск без установки пакета
    sys.path.insert(0, str(REPO_ROOT))

from rag_parser.chunkers.hybrid import HybridRagChunker  # noqa: E402
from rag_parser.parsers.dedoc_parser import DedocParser  # noqa: E402

INPUT_DIR = REPO_ROOT / "input"
ETALON_DIR = INPUT_DIR / "etalon"
OUTPUT_PATH = REPO_ROOT / "output" / "comparison_results.json"

ETALON_STEM = "gost_report"
FORMATS_TO_TEST: Tuple[str, ...] = (
    "gost_report.docx",
    "gost_report.doc",
    "gost_report.odt",
    "gost_report.rtf",
    "gost_report.pdf",
    "gost_report.html",
    "gost_report.htm",
    "gost_report.md",
    "gost_report.xml",
    "gost_report.txt",
)

#: Эталонный документ сравнивается сам с собой, дельт быть не должно.
REFERENCE_FORMAT = "gost_report.docx"

#: Длина превью заголовка в эталонном дереве (``TREE_TEXT_PREVIEW_LEN``).
#: Эталон сохранён с обрезанным текстом заголовков, поэтому сравниваем начала.
HEADER_PREVIEW_LEN = 30

CHUNK_SIZE = 200
OVERLAP = 50

#: Дельты, которые следуют из самого исходного файла, а не из кода.
#: Ключ — формат, значение — человекочитаемое описание ожидаемой дельты.
KNOWN_SOURCE_DELTAS: Dict[str, List[str]] = {
    "gost_report.md": [
        "в исходнике нет титульных строк «ПО ТЕХНИЧЕСКОМУ РЕГУЛИРОВАНИЮ И "
        "МЕТРОЛОГИИ» и «НАЦИОНАЛЬНЫЙ СТАНДАРТ РОССИЙСКОЙ ФЕДЕРАЦИИ» "
        "(эталонных заголовков 42, ожидается 40)",
    ],
    "gost_report.xml": [
        "в исходнике нет титульных строк «ПО ТЕХНИЧЕСКОМУ РЕГУЛИРОВАНИЮ И "
        "МЕТРОЛОГИИ» и «НАЦИОНАЛЬНЫЙ СТАНДАРТ РОССИЙСКОЙ ФЕДЕРАЦИИ»",
        "три строки заголовка лежат в одном <title> одной строкой без "
        "разделителей — они остаются одним заголовком",
    ],
    "gost_report.txt": [
        "исходник целиком в верхнем регистре — текст чанков сравнивается "
        "без учёта регистра",
    ],
    "gost_report.pdf": [
        "pdf2docx склеивает слова без пробелов («программнымспособом») и "
        "теряет часть титульного блока — ограничение конвертера",
    ],
}

#: Разрешённые расхождения по числу заголовков (эталон минус дельта источника).
#: Проверяется как «|факт - ожидание| <= tolerance», а не как точное число:
#: сверка идёт по выравненным заголовкам, лишние считаются ошибкой.
HEADER_COUNT_TOLERANCE: Dict[str, int] = {
    "gost_report.md": 2,
    "gost_report.xml": 4,
    "gost_report.txt": 3,
    "gost_report.pdf": 6,
}

CHUNK_COUNT_TOLERANCE: Dict[str, int] = {
    "gost_report.md": 20,
    "gost_report.xml": 25,
    "gost_report.txt": 45,
    "gost_report.pdf": 60,
}


# ---------------------------------------------------------------------------
# Нормализация текста
# ---------------------------------------------------------------------------
def normalize_text(text: str, *, ignore_case: bool = True) -> str:
    """Схлопывает пробелы и (по умолчанию) приводит текст к нижнему регистру.

    NFKC + ``ignore_case`` нужны, чтобы сравнение не ломалось на тире разной
    длины и на регистре (``.txt``-источник целиком в верхнем регистре).
    """
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = " ".join(normalized.split())
    return normalized.casefold() if ignore_case else normalized


def normalize_header(text: str) -> str:
    """Ключ заголовка: начало строки без ``…`` и без регистра.

    Эталонное дерево хранит превью заголовка (30 символов + ``…``), тогда как
    рабочее дерево — полный текст. Сравнивать beginnings, иначе любой длинный
    заголовок давал бы ложное «отсутствует».
    """
    return normalize_text(text)[:HEADER_PREVIEW_LEN]


# ---------------------------------------------------------------------------
# Обход дерева
# ---------------------------------------------------------------------------
def walk(node: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Все узлы дерева в порядке документа (pre-order)."""
    yield node
    for child in node.get("children", []) or []:
        yield from walk(child)


def headers_of(tree: Dict[str, Any]) -> List[Tuple[str, int]]:
    """``[(текст, уровень), ...]`` по всем заголовкам в порядке документа."""
    return [
        (node.get("text", ""), node.get("level", 0))
        for node in walk(tree)
        if node.get("type") == "header"
    ]


def count_tables(tree: Dict[str, Any]) -> int:
    return sum(1 for node in walk(tree) if node.get("type") == "table")


def _fingerprint(header: Tuple[str, int]) -> Tuple[str, int]:
    return normalize_header(header[0]), header[1]


def align_headers(
    etalon: Sequence[Tuple[str, int]], result: Sequence[Tuple[str, int]]
) -> List[str]:
    """Строки различий по заголовкам с выравниванием последовательностей.

    Сравнение по позиции («i-й заголовок против i-го») непригодно: одна
    потерянная строка сдвигает всё остальное, и тогда «не совпал уровень»
    Report получается для каждого заголовка после точки сдвига. Поэтому
    сравниваем как последовательности (``difflib``) и сообщаем только реальные
    расхождения: ``-`` потерян, ``+`` лишний, ``~`` отличается уровень.
    """
    differences: List[str] = []
    matcher = difflib.SequenceMatcher(
        a=[_fingerprint(item) for item in etalon], b=[_fingerprint(item) for item in result]
    )
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            for text, level in etalon[i1:i2]:
                differences.append(
                    f"заголовок отсутствует: L{level} «{text.strip()[:60]}»"
                )
            continue
        if tag == "insert":
            for text, level in result[j1:j2]:
                differences.append(
                    f"лишний заголовок: L{level} «{text.strip()[:60]}»"
                )
            continue
        # replace: часть совпала, часть отличается уровнем
        for offset in range(max(i2 - i1, j2 - j1)):
            left = etalon[i1 + offset] if i1 + offset < i2 else None
            right = result[j1 + offset] if j1 + offset < j2 else None
            if left is None:
                differences.append(
                    f"лишний заголовок: L{right[1]} «{right[0].strip()[:60]}»"
                )
            elif right is None:
                differences.append(
                    f"заголовок отсутствует: L{left[1]} «{left[0].strip()[:60]}»"
                )
            elif left[1] != right[1]:
                differences.append(
                    f"уровень «{left[0].strip()[:50]}»: эталон=L{left[1]}, "
                    f"факт=L{right[1]}"
                )
            else:
                differences.append(
                    f"заголовок отличается: эталон «{left[0].strip()[:60]}», "
                    f"факт «{right[0].strip()[:60]}»"
                )
    return differences


# ---------------------------------------------------------------------------
# Парсинг и чанкинг
# ---------------------------------------------------------------------------
def parse_and_chunk(
    file_path: Path, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP
) -> Dict[str, Any]:
    """Парсит файл и создаёт чанки (тот же пайплайн, что у API)."""
    parser = DedocParser()
    result = parser.parse(file_path)

    chunker = HybridRagChunker(
        chunk_size=chunk_size,
        overlap=overlap,
        source=file_path.name,
    )
    chunks = chunker.chunk(
        result.markdown,
        {"source": file_path.name, "format": result.format},
        tree=result.tree,
    )
    return {
        "tree": result.tree.to_dict(),
        "chunks": [chunk.to_dict() for chunk in chunks],
        "markdown": result.markdown,
        "metadata": result.metadata,
    }


# ---------------------------------------------------------------------------
# Сравнение
# ---------------------------------------------------------------------------
def compare_with_etalon(
    result: Dict[str, Any], etalon: Dict[str, Any], format_name: str
) -> List[str]:
    """Возвращает список различий (пустой список = паритет)."""
    differences: List[str] = []
    tolerance_headers = HEADER_COUNT_TOLERANCE.get(format_name, 0)
    tolerance_chunks = CHUNK_COUNT_TOLERANCE.get(format_name, 0)

    # 1. Заголовки: количество (с допуском источника) + выравненные уровни.
    etalon_headers = headers_of(etalon["tree"])
    result_headers = headers_of(result["tree"])
    delta = len(etalon_headers) - len(result_headers)
    if abs(delta) > tolerance_headers:
        differences.append(
            f"Заголовки: эталон={len(etalon_headers)}, факт={len(result_headers)}"
        )

    # 2. Уровни/тексты заголовков (с выравниванием, см. align_headers).
    differences.extend(align_headers(etalon_headers, result_headers))

    # 3. Таблицы.
    etalon_tables = count_tables(etalon["tree"])
    result_tables = count_tables(result["tree"])
    if etalon_tables != result_tables:
        differences.append(f"Таблицы: эталон={etalon_tables}, факт={result_tables}")

    # 4. Orphan-таблицы.
    etalon_orphan = (etalon["tree"].get("metadata") or {}).get("orphan_tables", 0)
    result_orphan = (result["tree"].get("metadata") or {}).get("orphan_tables", 0)
    if etalon_orphan != result_orphan:
        differences.append(f"Orphan таблицы: эталон={etalon_orphan}, факт={result_orphan}")

    # 5. Подавленные строки оглавления.
    etalon_toc = (etalon["tree"].get("metadata") or {}).get("toc_suppressed", 0)
    result_toc = (result["tree"].get("metadata") or {}).get("toc_suppressed", 0)
    if etalon_toc != result_toc:
        differences.append(
            f"TOC suppression: эталон={etalon_toc}, факт={result_toc}"
        )

    # 6. Чанки.
    etalon_chunks = etalon["chunks"]
    result_chunks = result["chunks"]
    delta = len(etalon_chunks) - len(result_chunks)
    if abs(delta) > tolerance_chunks:
        differences.append(
            f"Чанки: эталон={len(etalon_chunks)}, факт={len(result_chunks)}"
        )

    # 7. Breadcrumb: выравнивание по parent_id/порядку, чтобы оценивать
    #    структуру, а не смещение от разного числа заголовков.
    differences.extend(_compare_breadcrumbs(etalon_chunks, result_chunks))

    # 8. Тексты чанков (регистр источника нормализуем).
    differences.extend(_compare_chunk_texts(etalon_chunks, result_chunks))

    return differences


def _compare_breadcrumbs(
    etalon_chunks: Sequence[Dict[str, Any]], result_chunks: Sequence[Dict[str, Any]]
) -> List[str]:
    """Сравнение структуры разделов по последовательности крошек.

    Сравнивать крошку по ``parent_id`` нельзя: идентификаторы ``h_N``
    детерминированы порядком обхода, а число заголовков у разных форматов
    разное (у PDF, например, на 5 меньше), поэтому ``h_16`` у эталона и у
    факта — разные разделы и каждое такое сравнение даёт ложное расхождение.

    Поэтому сравниваем сами ПУТИ в порядке появления: у эталона должен быть
    тот же набор вложенностей («5 Общие положения» -> «5.1 Первое положение»),
    независимо от того, каким по счёту вышел заголовок.
    """
    etalon_paths = _breadcrumb_paths(etalon_chunks)
    result_paths = _breadcrumb_paths(result_chunks)
    if etalon_paths == result_paths:
        return []

    matcher = difflib.SequenceMatcher(
        a=[normalize_text(path or "") for path in etalon_paths],
        b=[normalize_text(path or "") for path in result_paths],
    )
    differences: List[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            differences.extend(
                f"нет разделов с крошкой «{(etalon_paths[index] or '∅')[:70]}»"
                for index in range(i1, i2)
            )
        elif tag == "insert":
            differences.extend(
                f"лишний раздел с крошкой «{(result_paths[index] or '∅')[:70]}»"
                for index in range(j1, j2)
            )
        else:
            differences.append(
                f"крошка раздела: эталон «{(etalon_paths[i1] or '∅')[:60]}», "
                f"факт «{(result_paths[j1] or '∅')[:60]}»"
            )
    return differences


def _breadcrumb_paths(chunks: Sequence[Dict[str, Any]]) -> List[Optional[str]]:
    """Крошки разделов в порядке первого появления (без повторов).

    Подряд идущие ``None`` (чанки вне разделов — титул, предисловие) схлопываются
    в один: сколько именно корневых заголовков у файла, структуру не описывает,
    а разные форматы теряют их по-разному и сравнение давало бы шум.
    """
    seen: set = set()
    paths: List[Optional[str]] = []
    for chunk in chunks:
        parent = chunk.get("parent_id")
        if parent is None or parent in seen:
            continue
        seen.add(parent)
        breadcrumb = chunk.get("breadcrumb")
        if breadcrumb is None and paths and paths[-1] is None:
            continue
        paths.append(breadcrumb)
    return paths


def _compare_chunk_texts(
    etalon_chunks: Sequence[Dict[str, Any]], result_chunks: Sequence[Dict[str, Any]]
) -> List[str]:
    """Сравнение текстов чанков: полные последовательности, первые 3 различия."""
    etalon_texts = [normalize_text(chunk.get("text", "")) for chunk in etalon_chunks]
    result_texts = [normalize_text(chunk.get("text", "")) for chunk in result_chunks]
    if etalon_texts == result_texts:
        return []

    matcher = difflib.SequenceMatcher(a=etalon_texts, b=result_texts)
    differences: List[str] = []
    changed = sum(
        max(i2 - i1, j2 - j1)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    )
    differences.append(
        f"Текст чанков отличается ({changed} шт. из {len(etalon_texts)})"
    )
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            differences.append(
                f"  чанк #{i1} отсутствует: «{_clip(etalon_chunks[i1])}»"
            )
        elif tag == "insert":
            differences.append(
                f"  лишний чанк #{j1}: «{_clip(result_chunks[j1])}»"
            )
        else:
            differences.append(
                f"  чанк #{i1}: эталон «{_clip(etalon_chunks[i1])}»"
            )
            differences.append(
                f"  чанк #{j1}: факт   «{_clip(result_chunks[j1])}»"
            )
        if len(differences) >= 8:
            differences.append("  … (сравнение прервано)")
            break
    return differences


def _clip(chunk: Dict[str, Any], limit: int = 90) -> str:
    text = normalize_text(chunk.get("text", ""), ignore_case=False)
    return text[:limit] + ("…" if len(text) > limit else "")


# ---------------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------------
def load_etalon(etalon_dir: Path = ETALON_DIR) -> Dict[str, Any]:
    """Читает эталонные артефакты (tree/chunks) из ``input/etalon/``.

    Отдельная папка обязательна: если положить ``gost_report.md`` рядом с
    исходниками, он ЗАМЕНИТ собой ``input/gost_report.md`` — а это входной
    файл одного из проверяемых форматов, и такой эталон сравнивал бы сам
    себя с собой (такое уже случалось).
    """
    tree_path = etalon_dir / f"{ETALON_STEM}.tree.json"
    chunks_path = etalon_dir / f"{ETALON_STEM}.chunks.json"
    markdown_path = etalon_dir / f"{ETALON_STEM}.md"
    missing = [
        str(path)
        for path in (tree_path, chunks_path, markdown_path)
        if not path.exists()
    ]
    if missing:
        raise SystemExit(
            "Нет эталонных артефактов: "
            + ", ".join(missing)
            + "\nСкопируйте gost_report.tree.json, gost_report.chunks.json и "
            "gost_report.md в input/etalon/ "
            "(их делает rag_parser/scripts/process_input_folder.py)."
        )
    return {
        "tree": json.loads(tree_path.read_text(encoding="utf-8")),
        "chunks": json.loads(chunks_path.read_text(encoding="utf-8")),
        "markdown": markdown_path.read_text(encoding="utf-8"),
    }


def format_report(results: Dict[str, Dict[str, Any]]) -> str:
    """Человекочитаемая таблица по всем форматам."""
    header = (
        f"{'формат':<22}{'заголовки':>12}{'таблицы':>9}"
        f"{'орфан':>7}{'чанки':>8}{'различий':>11}  статус"
    )
    lines = [header, "─" * len(header)]
    for name, info in results.items():
        if info.get("error"):
            lines.append(f"{name:<22}{'ОШИБКА':>12}{'':>9}{'':>7}{'':>8}{'':>11}  {info['error'][:60]}")
            continue
        stats = info["stats"]
        headers = f"{stats['headers']}/{stats['etalon_headers']}"
        tables = f"{stats['tables']}/{stats['etalon_tables']}"
        chunks = f"{stats['chunks']}/{stats['etalon_chunks']}"
        count = len(info["differences"])
        status = "OK" if count == 0 else f"{count} расхожд., {len(info['known_deltas'])} допущено"
        lines.append(
            f"{name:<22}{headers:>12}{tables:>9}{stats['orphan']:>7}"
            f"{chunks:>8}{count:>11}  {status}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scripts/compare_all_formats.py",
        description="Сравнение парсинга всех форматов с эталоном gost_report.docx",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=INPUT_DIR, help="папка с исходниками"
    )
    parser.add_argument(
        "--etalon-dir",
        type=Path,
        default=ETALON_DIR,
        help="папка с эталонными gost_report.{tree,chunks}.json и gost_report.md",
    )
    parser.add_argument(
        "--format",
        dest="formats",
        action="append",
        help="ограничить список форматов (можно указать несколько раз)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="считать известные дельты источника ошибками",
    )
    parser.add_argument(
        "--max-differences",
        type=int,
        default=None,
        help="код возврата 1, если у какого-либо формата различий больше",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH, help="куда писать JSON-отчёт"
    )
    args = parser.parse_args(argv)

    etalon = load_etalon(args.etalon_dir)
    formats = tuple(args.formats) if args.formats else FORMATS_TO_TEST

    results: Dict[str, Dict[str, Any]] = {}
    for filename in formats:
        file_path = args.input_dir / filename
        if not file_path.exists():
            print(f"⚠️  {filename} не найден, пропускаем")
            continue

        print(f"\n▶ {filename}")
        try:
            result = parse_and_chunk(file_path)
        except Exception as exc:  # noqa: BLE001 - формат мог не распарситься
            print(f"  ✖ ошибка парсинга: {type(exc).__name__}: {exc}")
            results[filename] = {"error": f"{type(exc).__name__}: {exc}"}
            continue

        differences = compare_with_etalon(result, etalon, filename)
        known = KNOWN_SOURCE_DELTAS.get(filename, [])
        tree = result["tree"]
        stats = {
            "headers": len(headers_of(tree)),
            "etalon_headers": len(headers_of(etalon["tree"])),
            "tables": count_tables(tree),
            "etalon_tables": count_tables(etalon["tree"]),
            "orphan": (tree.get("metadata") or {}).get("orphan_tables", 0),
            "chunks": len(result["chunks"]),
            "etalon_chunks": len(etalon["chunks"]),
            "toc_suppressed": (tree.get("metadata") or {}).get("toc_suppressed", 0),
        }
        results[filename] = {
            "stats": stats,
            "differences": differences,
            "known_deltas": known,
        }

        if differences:
            print(f"  ✖ {len(differences)} различий:")
            for line in differences[:12]:
                print(f"     - {line}")
            if len(differences) > 12:
                print(f"     … ещё {len(differences) - 12}")
        else:
            print("  ✔ совпадает с эталоном")
        for note in known:
            print(f"  ~ допущено ({filename}): {note}")

    print("\n" + format_report(results))

    payload = {
        "etalon": ETALON_STEM,
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nОтчёт: {args.output}")

    worst = 0
    for info in results.values():
        if info.get("error"):
            worst = max(worst, 99)
            continue
        count = len(info["differences"]) + (
            len(info["known_deltas"]) if args.strict else 0
        )
        worst = max(worst, count)
    if args.max_differences is not None and worst > args.max_differences:
        print(f"Превышен порог: {worst} > {args.max_differences}")
        return 1
    return 0 if worst == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())