from pathlib import Path
from typing import Any, List

from rag_parser.core.interfaces import BaseParser, DocumentParseError, ParseResult
from rag_parser.parsers.base import build_result, rows_to_markdown_table

SUPPORTED_EXTENSIONS: List[str] = [".xlsx", ".xls"]


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _forward_fill(rows: List[List[Any]]) -> List[List[Any]]:
    """Fills merged-cell gaps: calamine reports a value only in the top-left
    cell of a merged range, the rest come back empty. Empty cells inherit the
    nearest non-empty value from the left, then from above.
    """
    filled = [list(row) for row in rows]
    for row in filled:
        for index in range(1, len(row)):
            if _is_empty(row[index]) and not _is_empty(row[index - 1]):
                row[index] = row[index - 1]
    for row_index in range(1, len(filled)):
        above, current = filled[row_index - 1], filled[row_index]
        for index in range(len(current)):
            if _is_empty(current[index]) and index < len(above):
                if not _is_empty(above[index]):
                    current[index] = above[index]
    return filled


def _is_empty_sheet(rows: List[List[Any]]) -> bool:
    return all(all(_is_empty(value) for value in row) for row in rows)


class XlsxParser(BaseParser):
    format = "xlsx"
    extensions = (".xlsx", ".xls")

    def supported_extensions(self) -> List[str]:
        return list(SUPPORTED_EXTENSIONS)

    def parse(self, file_path: Path) -> ParseResult:
        from python_calamine import CalamineWorkbook

        path = Path(file_path)
        if not path.exists():
            raise DocumentParseError(f"File not found: {path}", filename=path.name)
        try:
            size_bytes = path.stat().st_size
        except OSError as exc:
            raise DocumentParseError(
                f"Cannot read file: {exc}", filename=path.name
            ) from exc
        if size_bytes == 0:
            raise DocumentParseError("File is empty", filename=path.name)

        extension = path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise DocumentParseError(
                f"Unsupported extension {extension!r} for XLSX parser. "
                f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
                filename=path.name,
            )

        try:
            workbook = CalamineWorkbook.from_path(str(path))
        except Exception as exc:  # noqa: BLE001 - calamine raises assorted errors
            raise DocumentParseError(
                f"Failed to open workbook: {exc}", filename=path.name
            ) from exc

        sections: List[str] = []
        total_rows = 0
        total_cols = 0
        sheet_count = 0
        try:
            names = list(workbook.sheet_names)
        except Exception as exc:  # noqa: BLE001
            raise DocumentParseError(
                f"Failed to read workbook sheets: {exc}", filename=path.name
            ) from exc

        for name in names:
            try:
                sheet = workbook.get_sheet_by_name(name)
                rows = sheet.to_python(skip_empty_area=False)
            except Exception as exc:  # noqa: BLE001
                raise DocumentParseError(
                    f"Failed to read sheet {name!r}: {exc}", filename=path.name
                ) from exc
            if not rows or _is_empty_sheet(rows):
                continue
            rows = _forward_fill(rows)
            table, num_rows, num_cols = rows_to_markdown_table(rows)
            if not table:
                continue
            sheet_count += 1
            total_rows += num_rows
            total_cols = max(total_cols, num_cols)
            sections.append(f"## Лист: {name}\n\n{table}")

        try:
            workbook.close()
        except Exception:  # noqa: BLE001 - best effort cleanup
            pass

        if not sections:
            raise DocumentParseError(
                "Workbook contains no readable sheets", filename=path.name
            )

        markdown = "\n\n".join(sections)
        result = build_result(
            path,
            self._detect_format(extension),
            markdown,
            {
                "format": self._detect_format(extension),
                "extension": extension,
                "size_bytes": size_bytes,
                "sheets": sheet_count,
                "sheet_names": names,
                "rows": total_rows,
                "columns": total_cols,
                "parser": "calamine",
            },
        )
        result.document_id = self._new_document_id()
        return result

    def _detect_format(self, extension: str) -> str:
        return "xls" if extension == ".xls" else "xlsx"
