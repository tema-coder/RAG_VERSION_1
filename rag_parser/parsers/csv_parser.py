from pathlib import Path

from rag_parser.core.interfaces import BaseParser, DocumentParseError, ParseResult
from rag_parser.parsers.base import (
    build_result,
    detect_dialect,
    parse_csv_rows,
    read_text_file,
    rows_to_markdown_table,
)


class CsvParser(BaseParser):
    format = "csv"
    extensions = (".csv", ".tsv")

    def parse(self, file_path: Path) -> ParseResult:
        path = Path(file_path)
        text, encoding, size_bytes = read_text_file(path)
        if not text.strip():
            raise DocumentParseError("CSV file is empty", filename=path.name)
        delimiter = detect_dialect(text[:8192])
        rows = parse_csv_rows(text, delimiter)
        if not rows:
            raise DocumentParseError("CSV file has no rows", filename=path.name)

        table, total_rows, columns = rows_to_markdown_table(rows)
        result = build_result(
            path,
            "csv",
            table,
            {
                "rows": total_rows,
                "data_rows": max(total_rows - 1, 0),
                "columns": columns,
                "delimiter": delimiter,
                "header": list(rows[0]) if rows else [],
                "size_bytes": size_bytes,
                "encoding": encoding,
            },
        )
        result.document_id = self._new_document_id()
        return result
