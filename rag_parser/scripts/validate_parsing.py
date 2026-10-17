"""Сквозная проверка парсинга и чанкинга против эталона (раздел 7 спеки).

Запуск:
    PYTHONPATH=. .venv/bin/python scripts/validate_parsing.py

Эталон (gost_report.*):
    - 11 ожидаемых заголовков с уровнями (см. spec 7.2)
    - 9 таблиц (для docx/pdf/html/md/xml/txt)
    - 0 orphan-таблиц для всех форматов кроме xlsx/csv
    - TOC-записи не должны стать заголовками (spec 7.6, test_txt < 56 headers)
    - Каждый табличный чанк = 1 строка данных + шапка (spec 5.4)
    - Breadcrumb у всех чанков (spec 5.2)
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Эталонные ожидания (раздел 7.2 спеки):
# (текст_заголовка, уровень)
ETALON_HEADERS_DOCX: List[Tuple[str, int]] = [
    ("1 Область применения", 1),
    ("5 Общие положения", 1),
    ("5.1 Первое положение", 2),
    ("5.2 Второе положение", 2),
    ("6 Структура документа", 1),
    ("6.1 Общие положения", 2),
    ("6.2 Элементы структуры", 2),
    ("6.2.1 Категории данных", 3),
    ("6.2.2 Квалификаторы идентификации", 3),
    ("6.2.3 Области использования", 3),
    ("7 Правила применения", 1),
]
ETALON_TABLES = 9  # spec 7.2: «Должно быть ровно 9 таблиц»
ETALON_ORPHAN = 0
ETALON_TOC_MAX_HEADERS = 55  # spec 7.6: «< 56 заголовков» для gost_report.txt

INPUT_DIR = Path(__file__).resolve().parent.parent / "input"

ALL_FILES = [
    "gost_report.docx",
    "gost_report.doc",
    "gost_report.pdf",
    "gost_report.md",
    "gost_report.txt",
    "gost_report.xml",
    "простая_html.html",
]


def iter_nodes(root):
    yield root
    for child in root.children:
        yield from iter_nodes(child)


def collect_headers(root):
    return [n for n in iter_nodes(root) if n.type == "header"]


def collect_tables(root):
    return [n for n in iter_nodes(root) if n.type == "table"]


def fmt_line(char: str = "─", n: int = 78) -> str:
    return char * n


def parse_file(parser, file_path: Path):
    print(f"\n{fmt_line('═')}")
    print(f"FILE: {file_path.name}")
    print(f"{fmt_line('═')}")
    try:
        result = parser.parse(file_path)
    except Exception as exc:
        print(f"  PARSE_ERROR: {type(exc).__name__}: {exc}")
        return None
    md_len = len(result.markdown)
    md_lines = result.markdown.count("\n") + 1
    print(f"  Markdown: {md_lines} lines, {md_len} chars")
    return result


def report_headers(root, label: str = ""):
    headers = collect_headers(root)
    levels = [h.level for h in headers]
    level_counts = {lvl: levels.count(lvl) for lvl in sorted(set(levels))}
    print(f"  Headers ({label}): total {len(headers)}")
    print(f"    By levels: {level_counts}")
    return headers


def report_tables(root):
    tables = collect_tables(root)
    orphan = root.metadata.get("orphan_tables", 0)
    print(f"  Tables: {len(tables)}, orphan: {orphan}")
    return tables, orphan


def check_docx_headers(headers):
    """Сверка с эталоном spec 7.2 для gost_report.docx."""
    actual = [(h.text.strip(), h.level) for h in headers]
    matched = []
    missing = []
    for etalon_text, etalon_level in ETALON_HEADERS_DOCX:
        # Поиск вхождения эталонного заголовка в фактический список
        found = None
        for actual_text, actual_level in actual:
            if etalon_text in actual_text and actual_level == etalon_level:
                found = (actual_text, actual_level)
                break
        if found:
            matched.append((etalon_text, etalon_level, found))
        else:
            missing.append((etalon_text, etalon_level))
    return matched, missing


def check_toc_suppression(headers):
    """spec 7.6: TOC-записи (с '...') не должны стать заголовками."""
    toc_leaked = [h for h in headers if "…" in h.text]
    return toc_leaked


def chunk_document(result, chunk_size=200, overlap=50):
    """Чанкинг документа через HybridRagChunker (FIXED strategy)."""
    from rag_parser.chunkers.hybrid import HybridRagChunker

    chunker = HybridRagChunker(
        text_strategy="token",
        chunk_size=chunk_size,
        overlap=overlap,
        source=result.filename,
    )
    metadata = {"source": result.filename, "format": result.format}
    chunks = chunker.chunk(result.markdown, metadata=metadata, tree=result.tree)
    return chunks


def report_chunks(chunks):
    table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
    text_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "text"]
    print(f"  Chunks: total {len(chunks)} (text: {len(text_chunks)}, tables: {len(table_chunks)})")
    # Проверка breadcrumb
    no_crumb = [c for c in chunks if not c.breadcrumb]
    print(f"    Without breadcrumb: {len(no_crumb)}")
    # Проверка табличных чанков: каждая = 1 строка данных (spec 5.4)
    bad_table_chunks = [
        c for c in table_chunks
        if c.metadata.get("data_rows", 0) != 1
    ]
    print(f"    Table chunks with data_rows != 1: {len(bad_table_chunks)}")
    # Примеры breadcrumb
    sample = chunks[:3] + chunks[-2:] if chunks else []
    for c in sample:
        kind = c.metadata.get("chunk_type", "?")
        crumb = c.breadcrumb or "(none)"
        text_preview = c.text[:60].replace("\n", " ")
        print(f"    [{kind}] crumb='{crumb}' | text='{text_preview}...'")
    return no_crumb, bad_table_chunks


def main():
    from rag_parser.parsers.dedoc_parser import DedocParser

    parser = DedocParser()
    summary = {}

    for filename in ALL_FILES:
        path = INPUT_DIR / filename
        if not path.exists():
            print(f"\nWARN: file not found: {path}")
            continue

        result = parse_file(parser, path)
        if result is None:
            summary[filename] = {"status": "PARSE_ERROR"}
            continue

        print(f"  --- Tree structure ---")
        headers = report_headers(result.tree, label=result.filename)
        tables, orphan = report_tables(result.tree)

        # Чанкинг
        print(f"  --- Chunking (Hybrid, token, size=200, overlap=50) ---")
        try:
            chunks = chunk_document(result)
            no_crumb, bad_tables = report_chunks(chunks)
        except Exception as exc:
            print(f"  CHUNK_ERROR: {type(exc).__name__}: {exc}")
            chunks, no_crumb, bad_tables = [], [], []

        # Специфичные проверки
        toc_leaked = check_toc_suppression(headers)

        # Сверка с эталоном для gost_report.*
        is_gost = "gost_report" in filename
        if is_gost and filename.endswith((".docx", ".doc")):
            matched, missing = check_docx_headers(headers)
            print(f"\n  --- spec 7.2 etalon check ---")
            print(f"    Etalon headers found: {len(matched)}/{len(ETALON_HEADERS_DOCX)}")
            if missing:
                print(f"    MISSING etalon headers:")
                for text, level in missing:
                    print(f"       - ({level}) {text}")
            else:
                print(f"    OK: all {len(ETALON_HEADERS_DOCX)} etalon headers in place")

            if len(tables) == ETALON_TABLES and orphan == ETALON_ORPHAN:
                print(f"    OK: tables={len(tables)} (etalon {ETALON_TABLES}), orphan={orphan} (etalon 0)")
            else:
                print(f"    BAD: tables={len(tables)} (etalon {ETALON_TABLES}), orphan={orphan} (etalon 0)")

        if is_gost and filename.endswith(".txt"):
            # spec 7.6: < 56 заголовков для gost_report.txt
            if len(headers) < ETALON_TOC_MAX_HEADERS + 1:
                print(f"\n  --- spec 7.6 TOC Suppression ---")
                print(f"    OK: headers={len(headers)} < {ETALON_TOC_MAX_HEADERS + 1} (etalon: < 56)")
            else:
                print(f"\n  --- spec 7.6 TOC Suppression ---")
                print(f"    BAD: headers={len(headers)} >= 56 (TOC not suppressed)")
            if toc_leaked:
                print(f"    TOC entries leaked into headers: {len(toc_leaked)}")
                for h in toc_leaked[:5]:
                    print(f"       - {h.text[:60]}")
            else:
                print(f"    OK: TOC entries (with '...') did not become headers")

        if is_gost and filename.endswith(".md"):
            # spec 7.3: ≥ 10 заголовков, ≥ 5 уровня 1, ≥ 3 уровня 2, ≥ 2 уровня 3
            levels = [h.level for h in headers]
            l1 = levels.count(1)
            l2 = levels.count(2)
            l3 = levels.count(3)
            print(f"\n  --- spec 7.3 MD parser ---")
            print(f"    Levels 1: {l1} (etalon >=5), 2: {l2} (>=3), 3: {l3} (>=2)")
            ok = len(headers) >= 10 and l1 >= 5 and l2 >= 3 and l3 >= 2
            print(f"    {'OK' if ok else 'BAD'}: headers={len(headers)} (etalon >=10)")
            if len(tables) == ETALON_TABLES and orphan == 0:
                print(f"    OK: tables={len(tables)}, orphan={orphan}")
            else:
                print(f"    BAD: tables={len(tables)} (etalon 9), orphan={orphan} (etalon 0)")

        summary[filename] = {
            "status": "OK",
            "headers": len(headers),
            "tables": len(tables),
            "orphan": orphan,
            "chunks": len(chunks),
            "no_crumb": len(no_crumb),
            "bad_table_chunks": len(bad_tables),
            "toc_leaked": len(toc_leaked),
        }

    # Итоговая сводка
    print(f"\n\n{fmt_line('=')}")
    print(f"SUMMARY")
    print(f"{fmt_line('=')}")
    hdr = f"{'File':<26} {'Status':<14} {'Hdrs':<5} {'Tbls':<5} {'Orph':<5} {'Chks':<5} {'NoCrmb':<7} {'BadTbl':<7} {'TOClk':<6}"
    print(hdr)
    print(fmt_line('-', len(hdr) + 5))
    for name, info in summary.items():
        if info["status"] != "OK":
            print(f"{name:<26} {info['status']}")
            continue
        print(
            f"{name:<26} {'OK':<14} "
            f"{info['headers']:<5} {info['tables']:<5} {info['orphan']:<5} "
            f"{info['chunks']:<5} {info['no_crumb']:<7} {info['bad_table_chunks']:<7} "
            f"{info['toc_leaked']:<6}"
        )


if __name__ == "__main__":
    sys.exit(main())
