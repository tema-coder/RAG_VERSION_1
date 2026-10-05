"""Generates a structurally complex DOCX fixture for the dedoc parser.

The document is built with python-docx and intentionally mixes everything the
parser must preserve: nested H1-H3 headings, multi-level bullet and numbered
lists, and tables with merged cells, multi-row headers and nested lists inside
cells.

Usage:
    python scripts/generate_test_doc.py [--output path/to/file.docx]
"""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent / "test_data" / "complex_structured.docx"
)


def _add_bullet_list(document, items, style="List Bullet") -> None:
    """Adds a bullet list; nested items are ``(text, [children])`` tuples."""
    for item in items:
        if isinstance(item, tuple):
            text, children = item
            document.add_paragraph(text, style=style)
            _add_bullet_list(document, children, style=_nested_style(style))
        else:
            document.add_paragraph(item, style=style)


def _add_numbered_list(document, items, style="List Number") -> None:
    """Adds a numbered list; nested items are ``(text, [children])`` tuples."""
    for item in items:
        if isinstance(item, tuple):
            text, children = item
            document.add_paragraph(text, style=style)
            _add_numbered_list(document, children, style=_nested_style(style))
        else:
            document.add_paragraph(item, style=style)


def _nested_style(style: str) -> str:
    """``List Bullet`` -> ``List Bullet 2`` -> ``List Bullet 3``."""
    head, _, level = style.rpartition(" ")
    return f"{head} {int(level) + 1}" if level.isdigit() else f"{style} 2"


def _add_nested_cell_list(cell, items) -> None:
    cell.paragraphs[0].text = items[0]
    for item in items[1:]:
        cell.add_paragraph(item, style="List Bullet")


def build_document():
    from docx import Document

    document = Document()

    document.add_heading("Техническое задание на систему RAG Parser", level=1)
    document.add_paragraph(
        "Документ описывает требования к сервису парсинга документов и "
        "используется как эталон для проверки извлечения структуры."
    )
    document.add_paragraph(
        "Вторая глава содержит списки, а третья — таблицы с объединёнными "
        "ячейками и вложенными списками."
    )

    document.add_heading("1. Общие сведения", level=2)
    document.add_paragraph(
        "Сервис принимает файл, конвертирует его в Markdown и возвращает "
        "структурированный ответ. Поддерживаются форматы PDF, DOCX, XLSX, "
        "HTML, TXT, MD и CSV."
    )
    _add_bullet_list(
        document,
        [
            (
                "Требования к функциональности",
                [
                    "Разбор структуры документа",
                    ("Заголовки H1-H3", ["Абзацы и списки"]),
                    "Таблицы с сохранением заголовков",
                ],
            ),
            (
                "Требования к производительности",
                ["Холодный старт не более 5 секунд", "Память не более 2 ГБ"],
            ),
        ],
    )

    document.add_heading("1.1 Ограничения", level=3)
    document.add_paragraph(
        "Сканированные PDF без текстового слоя не обрабатываются: OCR "
        "используется только в сторонних сервисах."
    )
    _add_numbered_list(
        document,
        [
            (
                "Этапы приёмки",
                [
                    "Установка зависимостей",
                    (
                        "Создание виртуального окружения",
                        ["Установка пакетов из requirements.txt"],
                    ),
                    "Запуск тестов",
                ],
            ),
            "Подготовка выгрузки",
        ],
    )

    document.add_page_break()

    document.add_heading("2. Спецификация чанкинга", level=2)
    document.add_paragraph(
        "Каждый тип чанкинга реализует свой алгоритм разбиения текста на "
        "фрагменты фиксированного размера."
    )
    _add_bullet_list(
        document,
        [
            (
                "Рекурсивный чанкинг",
                [
                    "Разделители: абзацы, строки, предложения, слова",
                    "Перекрытие реализуется поверх chonkie",
                ],
            ),
        ],
    )

    metrics = document.add_table(rows=4, cols=4)
    metrics.style = "Table Grid"
    # Two header rows with a merged cell: "Метрика" spans both header rows.
    metrics.cell(0, 0).merge(metrics.cell(1, 0)).text = "Метрика"
    metrics.cell(0, 1).merge(metrics.cell(0, 2)).text = "Значение"
    metrics.cell(0, 3).merge(metrics.cell(1, 3)).text = "Комментарий"
    metrics.cell(1, 1).text = "мин."
    metrics.cell(1, 2).text = "макс."
    metrics.cell(2, 0).text = "Размер чанка"
    metrics.cell(2, 1).text = "100"
    metrics.cell(2, 2).text = "2000"
    metrics.cell(2, 3).text = "Токенов cl100k_base"
    metrics.cell(3, 0).text = "Перекрытие"
    metrics.cell(3, 1).text = "0"
    metrics.cell(3, 2).text = "500"
    metrics.cell(3, 3).text = "Меньше размера чанка"

    formats = document.add_table(rows=4, cols=3)
    formats.style = "Table Grid"
    formats.cell(0, 0).text = "Формат"
    formats.cell(0, 1).text = "Парсер"
    formats.cell(0, 2).text = "Особенности"
    formats.cell(1, 0).text = "DOCX"
    formats.cell(1, 1).text = "MarkItDownParser"
    formats.cell(1, 2).text = "Структура, таблицы"
    formats.cell(2, 0).text = "XLSX"
    formats.cell(2, 1).text = "XlsxParser"
    formats.cell(2, 2).text = "Объединённые ячейки"
    formats.cell(3, 0).text = "CSV"
    formats.cell(3, 1).text = "CsvParser"
    formats.cell(3, 2).text = "Автодетект разделителя"
    # Nested list inside a cell.
    _add_nested_cell_list(
        formats.cell(1, 2),
        ["Структура", "Заголовки", "Таблицы"],
    )
    # Vertical merge: "Все форматы" spans two rows.
    formats.cell(2, 0).merge(formats.cell(3, 0))
    formats.cell(2, 0).text = "Все форматы"

    document.add_heading("2.1 Метрики качества", level=3)
    _add_numbered_list(
        document,
        [
            (
                "Полнота извлечения заголовков",
                ["Корректность Markdown-таблиц", "Проверка числа строк и столбцов"],
            ),
        ],
    )

    document.add_page_break()

    document.add_heading("3. Примеры запросов", level=2)
    document.add_paragraph(
        "Пример запуска API в фоне с сохранением результатов парсинга."
    )
    code = document.add_paragraph()
    code.add_run(
        "curl -X POST http://localhost:8000/api/parse_dedoc "
        "-F \"file=@complex_structured.docx\""
    ).font.name = "Courier New"

    document.add_heading("3.1 Ограничения версии 1.0", level=3)
    _add_bullet_list(
        document,
        [
            (
                "Хранилище результатов — в памяти процесса",
                ["LRU на 64 документа", "Не переживает перезапуск процесса"],
            ),
            "Аутентификация не реализована",
        ],
    )
    return document


def generate(output: Path = DEFAULT_OUTPUT) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    build_document().save(target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"path to the generated DOCX (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    target = generate(args.output)
    print(f"generated: {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())