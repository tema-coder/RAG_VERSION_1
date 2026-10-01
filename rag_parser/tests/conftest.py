from pathlib import Path

import pytest

TXT_UTF8 = "Первая строка документа.\nВторая строка документа.\nТретья строка.\n"
TXT_LATIN1 = "Café résumé naïve\nSecond line\n"
MD_DOC = (
    "# Заголовок первого уровня\n\n"
    "Абзац с текстом и **жирными** словами.\n\n"
    "## Подзаголовок\n\n"
    "- пункт один\n- пункт два\n\n"
    "| Колонка | Значение |\n|---|---|\n| A | 1 |\n"
)
CSV_SEMICOLON = "Колонка1;Колонка2;Колонка3\nЗначение1;Значение2;Значение3\n"
CSV_COMMA = "name,qty,price\napple,3,1.5\nbanana,10,0.75\n"
HTML_DOC = (
    "<!DOCTYPE html><html><head><title>T</title>"
    "<style>body{color:red}</style><script>var x=1;</script></head>"
    "<body><h1>Заголовок</h1><p>Текст со <a href='https://example.com'>ссылкой</a>.</p>"
    "<h2>Подзаголовок</h2><table><tr><th>A</th><th>B</th></tr>"
    "<tr><td>1</td><td>2</td></tr></table></body></html>"
)
HTML_MINIMAL = "<html><body><h3>Title</h3><ul><li>one</li><li>two</li></ul></body></html>"
SENTENCE_TEXT = (
    "Машинное обучение изучает алгоритмы, которые учатся на данных. "
    "Модели достигают высокой точности на проверочных наборах. "
    "Классические алгоритмы остаются интерпретируемыми и быстрыми. "
    "Глубокое обучение требует больших объёмов размеченных данных. "
    "Современные системы достигают уровня человека в ряде задач. "
) * 8


def write_bytes(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def make_text_files(tmp_path: Path) -> dict:
    return {
        "txt_utf8": write_bytes(tmp_path / "utf8.txt", TXT_UTF8.encode("utf-8")),
        "txt_latin1": write_bytes(tmp_path / "latin1.txt", TXT_LATIN1.encode("latin-1")),
        "md": write_bytes(tmp_path / "doc.md", MD_DOC.encode("utf-8")),
    }


def make_csv_files(tmp_path: Path) -> dict:
    return {
        "csv_semicolon": write_bytes(
            tmp_path / "semicolon.csv", CSV_SEMICOLON.encode("utf-8")
        ),
        "csv_comma": write_bytes(tmp_path / "comma.csv", CSV_COMMA.encode("utf-8")),
    }


def make_html_files(tmp_path: Path) -> dict:
    return {
        "html_full": write_bytes(tmp_path / "full.html", HTML_DOC.encode("utf-8")),
        "html_minimal": write_bytes(
            tmp_path / "minimal.html", HTML_MINIMAL.encode("utf-8")
        ),
    }


def make_docx_files(tmp_path: Path) -> dict:
    from docx import Document

    first = Document()
    first.add_heading("Заголовок документа", level=1)
    first.add_paragraph("Первый абзац текста.")
    first.add_heading("Второй раздел", level=2)
    first.add_paragraph("Второй абзац текста.")
    first_path = tmp_path / "first.docx"
    first.save(first_path)

    second = Document()
    second.add_paragraph("Абзац с таблицей.")
    table = second.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "A"
    table.cell(0, 1).text = "B"
    table.cell(1, 0).text = "1"
    table.cell(1, 1).text = "2"
    second_path = tmp_path / "second.docx"
    second.save(second_path)
    return {"docx_headings": first_path, "docx_table": second_path}


def make_xlsx_files(tmp_path: Path) -> dict:
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Данные"
    sheet.append(["Колонка1", "Колонка2"])
    sheet.append(["Значение1", 2])
    extra = workbook.create_sheet("Итоги")
    extra.append(["Показатель", "Значение"])
    extra.append(["Выручка", 100])
    first_path = tmp_path / "multi.xlsx"
    workbook.save(first_path)

    single = Workbook()
    only = single.active
    only.title = "Single"
    only.append(["id", "name"])
    for index in range(5):
        only.append([index, f"row-{index}"])
    second_path = tmp_path / "single.xlsx"
    single.save(second_path)

    merged = Workbook()
    sheet = merged.active
    sheet.title = "Data"
    sheet.append(["Report Title", None, None])
    sheet.merge_cells("A1:C1")
    sheet.append(["H1", "H2", "H3"])
    sheet.append(["a", "b", "c"])
    sheet.append(["merged-row", None, None])
    sheet.merge_cells("A4:C4")
    sheet.append(["d", "e", "f"])
    merged_path = tmp_path / "merged.xlsx"
    merged.save(merged_path)
    return {
        "xlsx_multi": first_path,
        "xlsx_single": second_path,
        "xlsx_merged": merged_path,
    }


def _pdf_escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _text_ops(lines, size=12):
    ops = []
    for index, line in enumerate(lines):
        if line:
            ops.append(
                f"BT /F1 {size} Tf 72 {720 - index * 20} Td ({_pdf_escape(line)}) Tj ET"
            )
    return ops


def _table_ops(table, x0=72, y0=400, cell_width=160, row_height=30):
    rows, columns = len(table), len(table[0])
    ops = ["0.5 w"]
    for row in range(rows + 1):
        ops.append(
            f"{x0} {y0 + row * row_height} m {x0 + columns * cell_width} {y0 + row * row_height} l S"
        )
    for column in range(columns + 1):
        ops.append(
            f"{x0 + column * cell_width} {y0} m {x0 + column * cell_width} {y0 + rows * row_height} l S"
        )
    for row, values in enumerate(table):
        for column, value in enumerate(values):
            position = y0 + rows * row_height - row * row_height - 18
            ops.append(
                f"BT /F1 11 Tf {x0 + column * cell_width + 8} {position} Td "
                f"({_pdf_escape(str(value))}) Tj ET"
            )
    return ops


def _build_pdf(pages, tables=None, path=None):
    """Builds a small valid PDF without external dependencies.

    The embedded base-14 Helvetica font covers latin text only, so PDF
    fixtures use ascii content.
    """
    operations = [list(page) for page in pages]
    for table in tables or []:
        operations.append(_table_ops(table))
    if not operations:
        operations = [[]]

    page_count = len(operations)
    first_content = 3 + page_count
    font_object = 3 + 2 * page_count
    objects = {}
    kids = " ".join(f"{3 + index} 0 R" for index in range(page_count))
    objects[1] = "<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>"
    for index, ops in enumerate(operations):
        objects[3 + index] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {first_content + index} 0 R "
            f"/Resources << /Font << /F1 {font_object} 0 R >> >> >>"
        )
        stream = "\n".join(ops)
        objects[first_content + index] = (
            f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"
        )
    objects[font_object] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += f"{number} 0 obj\n{objects[number]}\nendobj\n".encode("latin-1")
    xref_offset = len(out)
    size = max(objects) + 1
    out += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    for number in range(1, size):
        out += f"{offsets[number]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    return write_bytes(path, bytes(out))


def pdf_page(title, body=""):
    return _text_ops([title, *([body] if body else [])])


def make_pdf_files(tmp_path: Path) -> dict:
    return {
        "pdf_text": _build_pdf(
            [pdf_page("First page of the document."), pdf_page("Second page body.")],
            None,
            tmp_path / "text.pdf",
        ),
        "pdf_table": _build_pdf(
            [pdf_page("Quarterly report")],
            [[["Region", "Q1", "Q2"], ["North", "100", "120"], ["South", "90", "95"]]],
            tmp_path / "table.pdf",
        ),
    }


def make_invalid_files(tmp_path: Path) -> dict:
    return {
        "broken_pdf": write_bytes(tmp_path / "broken.pdf", b"%PDF-1.4\nnot really a pdf"),
        "empty_pdf": write_bytes(tmp_path / "empty.pdf", b""),
        "not_a_pdf": write_bytes(tmp_path / "fake.pdf", b"plain text file"),
        "empty_txt": write_bytes(tmp_path / "empty.txt", b""),
        "broken_docx": write_bytes(tmp_path / "broken.docx", b"not a zip container"),
        "broken_xlsx": write_bytes(tmp_path / "broken.xlsx", b"not a zip container"),
        "empty_csv": write_bytes(tmp_path / "empty.csv", b"\n"),
        "blank_csv": write_bytes(tmp_path / "blank.csv", b"   \n"),
        "empty_html": write_bytes(tmp_path / "empty.html", b""),
    }


@pytest.fixture
def text_files(tmp_path):
    return make_text_files(tmp_path)


@pytest.fixture
def csv_files(tmp_path):
    return make_csv_files(tmp_path)


@pytest.fixture
def html_files(tmp_path):
    return make_html_files(tmp_path)


@pytest.fixture
def docx_files(tmp_path):
    return make_docx_files(tmp_path)


@pytest.fixture
def xlsx_files(tmp_path):
    return make_xlsx_files(tmp_path)


@pytest.fixture
def pdf_files(tmp_path):
    return make_pdf_files(tmp_path)


@pytest.fixture
def invalid_files(tmp_path):
    return make_invalid_files(tmp_path)


@pytest.fixture
def sample_text():
    return SENTENCE_TEXT
