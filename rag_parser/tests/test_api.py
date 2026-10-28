import asyncio
import json
from pathlib import Path

import httpx
import pytest

from rag_parser.api.dependencies import (
    DocumentStore,
    get_chunker,
    get_parser,
    supported_extensions,
    validate_upload,
)
from rag_parser.api.schemas import ChunkingType
from rag_parser.core.config import settings
from rag_parser.core.interfaces import (
    DocumentParseError,
    FileTooLargeError,
    UnsupportedFormatError,
)
from rag_parser.main import app
from rag_parser.parsers.dedoc_parser import SUPPORTED_EXTENSIONS, DedocParser
from rag_parser.tests.conftest import _build_pdf, _text_ops

CHUNKING_TYPES = ["none", "fixed", "sentence", "semantic", "recursive"]


@pytest.fixture
def all_documents(pdf_files, docx_files, xlsx_files, xls_files, csv_files,
                  html_files, text_files):
    return {
        "txt": text_files["txt_utf8"],
        "csv": csv_files["csv_semicolon"],
        "tsv": csv_files["tsv"],
        "html": html_files["html_full"],
        "htm": html_files["htm"],
        "docx": docx_files["docx_headings"],
        "xlsx": xlsx_files["xlsx_multi"],
        "xls": xls_files["xls"],
        "pdf": pdf_files["pdf_table"],
    }


def upload(path, **data):
    return {"file": (path.name, path.read_bytes(), "application/octet-stream")}, data


def _mock_pdf_pipeline(monkeypatch):
    """Мокает Docling-пайплайн для тестов без фикстуры ``client``.

    Модели Docling в юнит-тестах не загружаются. ``client``-фикстура в
    ``conftest.py`` уже мокает ``run_pipeline``, но тесты, вызывающие
    ``process_function`` или ``httpx.AsyncClient`` напрямую, должны мокать
    отдельно. Реальная конвертация проверяется в ``test_docling_real.py``.
    """
    from rag_parser.parsers import pdf_pipeline
    from rag_parser.parsers.pdf_pipeline import PdfPipelineResult

    def _mock(path, **kwargs):
        from rag_parser.core.interfaces import DocumentParseError

        if path.suffix.lower() == ".pdf":
            try:
                data = path.read_bytes()
            except OSError as exc:
                raise DocumentParseError(
                    f"PDF pipeline mock: cannot read {path.name}: {exc}",
                    filename=path.name,
                ) from exc
            if b"%%EOF" not in data:
                raise DocumentParseError(
                    f"PDF pipeline mock: broken PDF {path.name}",
                    filename=path.name,
                )
        return PdfPipelineResult(
            markdown=f"Docling pipeline mocked: {path.name}",
            metadata={
                "source_pipeline": "docling",
                "input_format": "pdf",
                "pages": 1,
                "format": "dedoc",
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "parser": "docling",
            },
        )

    monkeypatch.setattr(pdf_pipeline, "run_pipeline", _mock)


def _hub_reachable() -> bool:
    import socket

    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
        return True
    except OSError:
        return False


def _walk_tree(node):
    yield node
    for child in node.get("children", []):
        yield from _walk_tree(child)


class TestMetaEndpoints:
    def test_health(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_formats(self, client):
        response = client.get("/api/formats")
        assert response.status_code == 200
        body = response.json()
        for extension in SUPPORTED_EXTENSIONS:
            assert extension in body["supported_extensions"]
        assert ".pptx" not in body["supported_extensions"]
        assert ".epub" not in body["supported_extensions"]
        assert len(body["supported_extensions"]) == len(SUPPORTED_EXTENSIONS)
        assert body["max_file_size_mb"] == settings.MAX_FILE_SIZE_MB
        assert body["chunking_types"] == CHUNKING_TYPES

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["supported_extensions"]

    def test_openapi(self, client):
        assert client.get("/openapi.json").status_code == 200


class TestParse:
    @pytest.mark.parametrize(
        "kind", ["txt", "csv", "tsv", "html", "htm", "docx", "xlsx", "xls", "pdf"]
    )
    def test_all_formats(self, client, all_documents, kind):
        path = all_documents[kind]
        files, data = upload(path)
        response = client.post("/api/parse", files=files, data=data)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["filename"] == path.name
        assert body["format"] == "dedoc"
        assert body["document_id"]
        assert body["markdown"].strip()
        assert body["chunks"] is None
        # PDF идёт через Docling-пайплайн, остальные — через dedoc
        # напрямую. Проверяем только, что парсер определён.
        assert body["metadata"]["parser"] in ("dedoc", "docling")
        assert body["tree"] is not None
        assert body["tree"]["type"] == "document"
        assert body["tree"]["children"]

    def test_tree_shape(self, client, docx_files):
        files, data = upload(docx_files["docx_headings"])
        body = client.post("/api/parse", files=files, data=data).json()
        headers = [n for n in _walk_tree(body["tree"]) if n["type"] == "header"]
        assert headers
        for header in headers:
            assert header["level"] >= 1
            assert len(header["text"]) <= 31
            assert "annotations" not in header["metadata"]

    def test_pdf_markdown_metadata(self, client, pdf_files):
        files, data = upload(pdf_files["pdf_table"])
        response = client.post("/api/parse", files=files, data=data)
        assert response.status_code == 200
        body = response.json()
        assert body["format"] == "dedoc"
        assert body["pages"] == 1  # Docling pipeline metadata
        assert body["metadata"]["format"] == "dedoc"
        # PDF идёт через Docling-пайплайн, остальные — через dedoc.
        assert body["metadata"]["parser"] in ("dedoc", "docling")
        assert body["metadata"]["size_bytes"] > 0
        # В тестах Docling-пайплайн замокан, поэтому содержимое markdown
        # не проверяем — только что он непустой. Реальная конвертация
        # проверяется в ``test_docling_real.py``.
        assert body["markdown"].strip()

    @pytest.mark.parametrize("extension", ["png", "jpg", "jpeg"])
    def test_image_upload_uses_docling(self, client, tmp_path, extension):
        from PIL import Image

        path = tmp_path / f"scan.{extension}"
        Image.new("RGB", (40, 20), "white").save(path)
        files, data = upload(path)
        response = client.post("/api/parse", files=files, data=data)

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["metadata"]["input_format"] == "image"
        assert body["metadata"]["extension"] == f".{extension}"
        assert body["metadata"]["parser"] == "docling"
        assert body["markdown"].strip()

    @pytest.mark.parametrize("chunking_type", CHUNKING_TYPES)
    def test_chunking_types(self, client, pdf_files, chunking_type):
        if chunking_type == "semantic" and not _hub_reachable():
            pytest.skip("semantic chunking needs the embedding model")
        files, data = upload(
            pdf_files["pdf_table"],
            chunking_type=chunking_type,
            chunk_size="100",
            overlap="10",
        )
        response = client.post("/api/parse", files=files, data=data)
        assert response.status_code == 200, response.text
        chunks = response.json()["chunks"]
        if chunking_type == "none":
            assert chunks is None
        else:
            expected = {
                "fixed": "token",
                "sentence": "sentence",
                "semantic": "semantic",
                "recursive": "recursive",
            }[chunking_type]
            assert chunks
            for index, chunk in enumerate(chunks):
                assert chunk["index"] == index
                assert chunk["text"].strip()
                assert chunk["tokens"] > 0
                assert chunk["source"] == "table.pdf"
                assert chunk["metadata"]["chunk_type"] in ("text", "table")
            text_chunks = [
                chunk
                for chunk in chunks
                if chunk["metadata"]["chunk_type"] == "text"
            ]
            table_chunks = [
                chunk
                for chunk in chunks
                if chunk["metadata"]["chunk_type"] == "table"
            ]
            assert text_chunks
            assert all(
                chunk["metadata"]["chunker"] == expected for chunk in text_chunks
            )
            # PDF без Java/tesseract не даёт структурированных таблиц —
            # табличные чанки проверяются на xlsx/docx ниже.
            for chunk in table_chunks:
                assert chunk["metadata"]["chunker"] == "table"
                assert chunk["text_for_embedding"]
                assert "|" not in chunk["text_for_embedding"]

    def test_breadcrumb_enrichment(self, client, docx_files, xlsx_files):
        files, data = upload(
            docx_files["docx_headings"], chunking_type="fixed", chunk_size="100"
        )
        chunks = client.post("/api/parse", files=files, data=data).json()["chunks"]
        assert chunks
        enriched = [c for c in chunks if c["breadcrumb"]]
        assert enriched
        for chunk in enriched:
            assert chunk["text_for_embedding"].startswith(f"[{chunk['breadcrumb']}] ")
        xfiles, xdata = upload(
            xlsx_files["xlsx_multi"], chunking_type="fixed", chunk_size="100"
        )
        xchunks = client.post("/api/parse", files=xfiles, data=xdata).json()["chunks"]
        tables = [c for c in xchunks if c["metadata"].get("chunk_type") == "table"]
        assert tables
        for chunk in tables:
            assert chunk["text_for_embedding"]
            assert "|" not in chunk["text_for_embedding"]

    def test_table_breadcrumb_prefix(self, client, complex_docx):
        files, data = upload(complex_docx, chunking_type="fixed", chunk_size="100")
        body = client.post("/api/parse", files=files, data=data).json()
        tables = [
            c for c in body["chunks"] if c["metadata"].get("chunk_type") == "table"
        ]
        assert tables
        for chunk in tables:
            if chunk["breadcrumb"]:
                assert chunk["text"].startswith(f"[{chunk['breadcrumb']}] ")
                assert "|" not in chunk["text_for_embedding"]

    def test_table_under_subsection_breadcrumb(self, client, tmp_path):
        """Таблица сразу за 2.1 → крошка «Раздел 2 > 2.1»."""
        from docx import Document

        doc = Document()
        doc.add_paragraph("Раздел 2. Сведения об обществе и его работе")
        doc.add_paragraph("2.1. Реквизиты общества с ограниченной ответственностью")
        table = doc.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Поле"
        table.cell(0, 1).text = "Значение"
        table.cell(1, 0).text = "ОГРН"
        table.cell(1, 1).text = "123"
        doc.add_paragraph("Хвост документа после таблицы.")
        path = tmp_path / "sub.docx"
        doc.save(path)
        files, data = upload(path, chunking_type="fixed", chunk_size="100")
        body = client.post("/api/parse", files=files, data=data).json()
        tables = [
            c for c in body["chunks"] if c["metadata"].get("chunk_type") == "table"
        ]
        assert tables
        for chunk in tables:
            assert chunk["breadcrumb"] is not None
            assert chunk["breadcrumb"].startswith("Раздел 2. Сведения")
            assert "2.1" in chunk["breadcrumb"]
            assert "|" not in chunk["text_for_embedding"]

    def test_header_recognition_is_deterministic(self, client, header_docs):
        """Порога нет: два разбора одного файла дают одинаковое дерево.

        Раньше здесь сравнивались пороги 0.0 и 8.0; теперь классификация
        детерминированная (7 жёстких правил), поэтому проверяем, что
        повторный разбор не меняет результат.
        """
        files_1, data_1 = upload(header_docs["unstyled"])
        files_2, data_2 = upload(header_docs["unstyled"])
        first = client.post("/api/parse", files=files_1, data=data_1).json()
        second = client.post("/api/parse", files=files_2, data=data_2).json()
        assert first["markdown"] == second["markdown"]
        assert [n["text"] for n in _walk_tree(first["tree"])] == [
            n["text"] for n in _walk_tree(second["tree"])
        ]

    def test_header_threshold_parameter_ignored(self, client, header_docs):
        """Переданный header_threshold не ломает запрос и игнорируется."""
        files_with, data_with = upload(header_docs["unstyled"], header_threshold="8.0")
        files_without, data_without = upload(header_docs["unstyled"])
        with_param = client.post("/api/parse", files=files_with, data=data_with).json()
        plain = client.post("/api/parse", files=files_without, data=data_without).json()
        assert with_param["markdown"] == plain["markdown"]

    def test_table_chunks_carry_text_for_embedding(self, client, xlsx_files):
        files, data = upload(
            xlsx_files["xlsx_multi"], chunking_type="fixed", chunk_size="100"
        )
        chunks = client.post("/api/parse", files=files, data=data).json()["chunks"]
        table_chunks = [
            chunk
            for chunk in chunks
            if chunk["metadata"].get("chunk_type") == "table"
        ]
        assert table_chunks
        for chunk in table_chunks:
            assert chunk["text_for_embedding"]
            assert "|" not in chunk["text_for_embedding"]
            assert ":" in chunk["text_for_embedding"]

    def test_defaults_without_form_fields(self, client, text_files):
        files, data = upload(text_files["txt_utf8"])
        response = client.post("/api/parse", files=files)
        assert response.status_code == 200
        assert response.json()["chunks"] is None

    def test_missing_file_field(self, client):
        response = client.post("/api/parse", data={"chunking_type": "none"})
        assert response.status_code == 400

    def test_parse_dedoc_removed(self, client, docx_files):
        files, data = upload(docx_files["docx_headings"])
        assert client.post("/api/parse_dedoc", files=files, data=data).status_code == 404


class TestParseErrors:
    def test_unsupported_extension(self, client, tmp_path):
        path = tmp_path / "virus.exe"
        path.write_bytes(b"MZ\x90\x00binary")
        files, data = upload(path)
        response = client.post("/api/parse", files=files, data=data)
        assert response.status_code == 422
        assert "Unsupported" in response.json()["detail"]

    def test_pptx_rejected_with_422(self, client, tmp_path):
        path = tmp_path / "slides.pptx"
        path.write_bytes(b"fake word content")
        files, data = upload(path)
        response = client.post("/api/parse", files=files, data=data)
        assert response.status_code == 422

    def test_broken_pdf(self, client, invalid_files):
        files, data = upload(invalid_files["broken_pdf"])
        response = client.post("/api/parse", files=files, data=data)
        assert response.status_code == 400
        assert response.headers.get("X-Error-Type") == "parse_error"

    def test_fake_pdf(self, client, invalid_files):
        files, data = upload(invalid_files["not_a_pdf"])
        response = client.post("/api/parse", files=files, data=data)
        assert response.status_code == 400

    def test_empty_file(self, client, invalid_files):
        files, data = upload(invalid_files["empty_pdf"])
        response = client.post("/api/parse", files=files, data=data)
        assert response.status_code == 400

    def test_broken_docx(self, client, invalid_files):
        files, data = upload(invalid_files["broken_docx"])
        response = client.post("/api/parse", files=files, data=data)
        assert response.status_code == 400

    def test_mime_mismatch_for_office(self, client, tmp_path):
        path = tmp_path / "fake.docx"
        path.write_bytes(b"just plain text, definitely not a zip")
        files, data = upload(path)
        response = client.post("/api/parse", files=files, data=data)
        assert response.status_code == 400

    def test_file_too_large(self, client, text_files, monkeypatch):
        monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 0)
        files, data = upload(text_files["txt_utf8"])
        response = client.post("/api/parse", files=files, data=data)
        assert response.status_code == 413
        assert response.headers.get("X-Error-Type") == "file_too_large"

    @pytest.mark.parametrize(
        "params",
        [
            {"chunk_size": "10"},
            {"chunk_size": "5000"},
            {"overlap": "900"},
            {"chunk_size": "100", "overlap": "100"},
            {"chunking_type": "unknown"},
            {"threshold": "0"},
            {"threshold": "1.5"},
        ],
    )
    def test_invalid_chunk_parameters(self, client, text_files, params):
        files, data = upload(text_files["txt_utf8"], **params)
        response = client.post("/api/parse", files=files, data=data)
        assert response.status_code == 400

    def test_internal_error_is_500(self, client, text_files, monkeypatch):
        import rag_parser.api.routes as routes

        class Boom:
            format = "text"

            def parse(self, file_path):
                raise RuntimeError("boom")

        monkeypatch.setattr(routes, "get_parser", lambda filename, **kw: Boom())
        files, data = upload(text_files["txt_utf8"])
        response = client.post("/api/parse", files=files, data=data)
        assert response.status_code == 500
        assert "boom" in response.json()["detail"]


class TestDownload:
    def test_download_markdown(self, client, pdf_files):
        files, data = upload(pdf_files["pdf_table"], chunking_type="sentence", chunk_size="100")
        document_id = client.post("/api/parse", files=files, data=data).json()["document_id"]
        response = client.get(f"/api/parse/{document_id}/download")
        assert response.status_code == 200
        assert "table.pdf.md" in response.headers["content-disposition"]
        # В тестах PDF-пайплайн замокан — проверяем только что markdown
        # непустой, а не конкретное содержимое.
        assert response.text.strip()

    def test_download_json(self, client, xlsx_files):
        files, data = upload(xlsx_files["xlsx_multi"], chunking_type="fixed", chunk_size="100")
        document_id = client.post("/api/parse", files=files, data=data).json()["document_id"]
        response = client.get(f"/api/parse/{document_id}/download?format=json")
        assert response.status_code == 200
        payload = json.loads(response.content)
        assert payload["document_id"] == document_id
        assert payload["format"] == "dedoc"
        assert payload["tree"] is not None
        assert payload["chunks"]
        assert payload["chunks"][0]["metadata"]["chunk_type"] in ("text", "table")

    def test_download_tree(self, client, docx_files):
        files, data = upload(docx_files["docx_headings"])
        document_id = client.post("/api/parse", files=files, data=data).json()["document_id"]
        response = client.get(f"/api/parse/{document_id}/download?format=tree")
        assert response.status_code == 200
        payload = json.loads(response.content)
        assert payload["type"] == "document"
        assert payload["children"]

    def test_download_unknown_document(self, client):
        assert client.get("/api/parse/does-not-exist/download").status_code == 404

    def test_download_invalid_format(self, client, text_files):
        files, data = upload(text_files["txt_utf8"])
        document_id = client.post("/api/parse", files=files, data=data).json()["document_id"]
        assert client.get(f"/api/parse/{document_id}/download?format=docx").status_code == 400

    def test_store_is_bounded(self, client, text_files):
        files, data = upload(text_files["txt_utf8"])
        for _ in range(6):
            document_id = client.post("/api/parse", files=files, data=data).json()["document_id"]
        assert client.get(f"/api/parse/{document_id}/download").status_code == 200


class TestEndToEnd:
    def test_upload_parse_chunk_download(self, client, tmp_path):
        path = _build_pdf(
            [_text_ops([f"Page {index} of the document."]) for index in range(1, 6)],
            [[["Item", "Amount"], ["Nails", "100"], ["Boards", "250"]]],
            tmp_path / "e2e.pdf",
        )
        files, data = upload(path, chunking_type="fixed", chunk_size="120", overlap="10")
        parse_response = client.post("/api/parse", files=files, data=data)
        assert parse_response.status_code == 200
        body = parse_response.json()

        assert body["format"] == "dedoc"
        # В тестах PDF-пайплайн замокан — содержимое не проверяем, только
        # что markdown непустой и чанки построены.
        assert body["markdown"].strip()
        assert body["chunks"]
        assert body["tree"] is not None
        kinds = set()
        for index, chunk in enumerate(body["chunks"]):
            assert chunk["index"] == index
            assert chunk["tokens"] <= 128
            kinds.add(chunk["metadata"]["chunk_type"])
        assert kinds <= {"text", "table"}

        document_id = body["document_id"]
        markdown = client.get(f"/api/parse/{document_id}/download?format=md")
        assert markdown.status_code == 200
        assert markdown.text == body["markdown"]

        payload = client.get(f"/api/parse/{document_id}/download?format=json").json()
        assert payload["markdown"] == body["markdown"]
        assert len(payload["chunks"]) == len(body["chunks"])
        assert payload["metadata"]["parser"] in ("dedoc", "docling")

    def test_csv_table_routes_to_table_chunker(self, client, csv_files):
        files, data = upload(
            csv_files["csv_comma"], chunking_type="fixed", chunk_size="100"
        )
        response = client.post("/api/parse", files=files, data=data)
        assert response.status_code == 200, response.text
        chunks = response.json()["chunks"]
        assert chunks
        assert all(
            chunk["metadata"]["chunk_type"] == "table" for chunk in chunks
        )
        for chunk in chunks:
            assert chunk["metadata"]["chunker"] == "table"
            assert chunk["text"].splitlines()[0].startswith("| name")
            assert chunk["text_for_embedding"]
            assert "|" not in chunk["text_for_embedding"]
            assert "name:" in chunk["text_for_embedding"]

    def test_ten_page_pdf_performance(self, client, tmp_path):
        path = _build_pdf(
            [_text_ops([f"Page {index}. Body of page {index}."]) for index in range(1, 11)],
            None,
            tmp_path / "ten.pdf",
        )
        files, data = upload(path, chunking_type="fixed", chunk_size="500", overlap="50")
        import time

        started = time.perf_counter()
        response = client.post("/api/parse", files=files, data=data)
        elapsed = time.perf_counter() - started
        assert response.status_code == 200
        # Содержимое не проверяем — PDF-пайплайн замокан. Только что
        # markdown построен и время разумное.
        assert response.json()["markdown"].strip()
        assert elapsed < 60.0, f"parsing took {elapsed:.2f}s"


class TestDependenciesUnit:
    def test_get_parser_by_extension(self):
        for extension in SUPPORTED_EXTENSIONS:
            assert isinstance(get_parser(f"file{extension}"), DedocParser)

    def test_get_parser_case_insensitive(self):
        assert isinstance(get_parser("file.PDF"), DedocParser)

    def test_get_parser_unsupported(self):
        with pytest.raises(UnsupportedFormatError):
            get_parser("file.exe")
        with pytest.raises(UnsupportedFormatError):
            get_parser("file.pptx")

    def test_get_parser_has_no_threshold(self):
        """Порога заголовков больше нет: get_parser(filename) — без параметров."""
        import inspect

        params = inspect.signature(get_parser).parameters
        assert list(params) == ["filename"]
        assert isinstance(get_parser("a.docx"), DedocParser)

    def test_get_chunker_none(self):
        assert get_chunker(ChunkingType.NONE) is None

    @pytest.mark.parametrize(
        "chunking_type", [ChunkingType.FIXED, ChunkingType.SENTENCE, ChunkingType.SEMANTIC]
    )
    def test_get_chunker(self, chunking_type):
        from rag_parser.chunkers.hybrid import HybridRagChunker

        chunker = get_chunker(chunking_type, 200, 20)
        assert isinstance(chunker, HybridRagChunker)
        if chunking_type == ChunkingType.SEMANTIC:
            pytest.skip("semantic chunking needs the embedding model")
        assert chunker.chunk("Тест. Ещё тест.", {})[0].tokens > 0

    def test_get_chunker_overlap_validation(self):
        with pytest.raises(ValueError):
            get_chunker(ChunkingType.FIXED, 100, 100)

    def test_supported_extensions(self):
        extensions = supported_extensions()
        assert sorted(extensions) == sorted(SUPPORTED_EXTENSIONS)
        assert ".xlsm" not in extensions
        assert ".pptx" not in extensions
        assert ".md" in extensions          # добавлен в Этапе 1
        assert ".doc" in extensions
        assert ".odt" in extensions
        assert ".rtf" in extensions
        assert ".xml" in extensions

    def test_validate_upload_size(self, monkeypatch):
        monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 0)
        with pytest.raises(FileTooLargeError):
            validate_upload("a.txt", b"x" * 10)

    def test_validate_upload_empty(self):
        with pytest.raises(DocumentParseError, match="empty"):
            validate_upload("a.txt", b"")

    def test_validate_upload_extension(self):
        with pytest.raises(UnsupportedFormatError):
            validate_upload("a.exe", b"MZ")

    def test_validate_upload_without_filename(self):
        with pytest.raises(DocumentParseError):
            validate_upload("", b"data")

    def test_document_store_lru(self):
        store = DocumentStore(max_items=2)
        first = store.add({"document_id": "one", "markdown": "a"})
        store.add({"document_id": "two", "markdown": "b"})
        store.add({"document_id": "three", "markdown": "c"})
        assert store.get(first) is None
        assert store.get("three")["markdown"] == "c"
        assert len(store) == 2
        store.clear()
        assert len(store) == 0


class TestAsyncClient:
    async def test_parse_with_httpx(self, text_files, tmp_path, monkeypatch):
        monkeypatch.setattr(
            settings, "DEDOC_OUTPUT_DIR", str(tmp_path / "output" / "dedoc_results")
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as async_client:
            health = await async_client.get("/api/health")
            assert health.status_code == 200

            path = text_files["txt_utf8"]
            files = {"file": (path.name, path.read_bytes(), "text/plain")}
            response = await async_client.post(
                "/api/parse", files=files, data={"chunking_type": "sentence", "chunk_size": "100"}
            )
            assert response.status_code == 200
            document_id = response.json()["document_id"]
            assert response.json()["tree"] is not None
            download = await async_client.get(
                f"/api/parse/{document_id}/download?format=json"
            )
            assert download.status_code == 200
            assert download.json()["document_id"] == document_id

    async def test_concurrent_requests(self, pdf_files, tmp_path, monkeypatch):
        monkeypatch.setattr(
            settings, "DEDOC_OUTPUT_DIR", str(tmp_path / "output" / "dedoc_results")
        )
        # PDF-пайплайн замокан (Docling не установлен в тестах).
        _mock_pdf_pipeline(monkeypatch)
        transport = httpx.ASGITransport(app=app)
        path = pdf_files["pdf_text"]
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as async_client:
            files = {"file": (path.name, path.read_bytes(), "application/pdf")}
            responses = await asyncio.gather(
                *[async_client.post("/api/parse", files=files) for _ in range(5)]
            )
        assert [response.status_code for response in responses] == [200] * 5
        assert len({response.json()["document_id"] for response in responses}) == 5


class TestFrontend:
    def test_process_function(self, pdf_files, tmp_path, monkeypatch):
        monkeypatch.setattr(
            settings, "DEDOC_OUTPUT_DIR", str(tmp_path / "output" / "dedoc_results")
        )
        # PDF-пайплайн замокан (Docling не установлен в тестах).
        _mock_pdf_pipeline(monkeypatch)
        from rag_parser.frontend.app import process

        class Upload:
            name = pdf_files["pdf_text"].name

            def __init__(self, data):
                self._data = data

            def getbuffer(self):
                return self._data

        result = process(
            Upload(pdf_files["pdf_text"].read_bytes()),
            ChunkingType.SENTENCE,
            200,
            20,
        )
        assert result["format"] == "dedoc"
        assert result["markdown"]
        assert result["chunks"]
        assert result["document_id"]
        assert result["tree"] is not None

    def test_process_raises_document_error(self, invalid_files, monkeypatch):
        # PDF-пайплайн замокан так, что битый PDF кидает DocumentParseError
        # (PyMuPDF не может открыть). Мок нужен, т.к. process() вызывает
        # parser.parse() напрямую, а Docling не установлен.
        _mock_pdf_pipeline(monkeypatch)
        from rag_parser.frontend.app import process

        class Upload:
            name = "broken.pdf"

            def getbuffer(self):
                return invalid_files["broken_pdf"].read_bytes()

        with pytest.raises(DocumentParseError):
            process(Upload(), ChunkingType.NONE, 500, 50)


class TestParentMetadataInApi:
    """parent_id и chunk_order должны доезжать до ответа POST /api/parse."""

    def test_response_contains_parent_fields(self, client, docx_files):
        files, data = upload(docx_files["docx_headings"],
                             chunking_type="fixed")
        response = client.post("/api/parse", files=files, data=data)
        assert response.status_code == 200
        chunks = response.json()["chunks"] or []
        assert chunks
        for chunk in chunks:
            assert "parent_id" in chunk
            assert "chunk_order" in chunk
            assert isinstance(chunk["chunk_order"], int)

    def test_chunked_response_groups_by_parent(self, client, docx_files):
        files, data = upload(docx_files["docx_headings"],
                             chunking_type="fixed")
        response = client.post("/api/parse", files=files, data=data)
        chunks = response.json()["chunks"] or []
        assert chunks
        groups = {}
        for chunk in chunks:
            if chunk["parent_id"]:
                groups.setdefault(chunk["parent_id"], []).append(
                    chunk["chunk_order"]
                )
        assert groups, "в документе с заголовками должны быть parent_id"
        for parent_id, orders in groups.items():
            assert orders == list(range(len(orders))), parent_id


class TestFrontendAcceptedTypes:
    """Загрузчик UI не должен расходиться с бэкендом по списку форматов.

    Регрессия: в app.py был захардкожен список из 9 расширений, из-за чего
    Streamlit отбрасывал .doc/.odt/.rtf/.xml/.md, хотя парсер их читает.
    """

    def test_accepts_all_supported_extensions(self):
        from rag_parser.core.config import DEFAULT_SUPPORTED_EXTENSIONS
        from rag_parser.frontend.app import ACCEPTED_TYPES

        assert ACCEPTED_TYPES == [
            ext.lstrip(".").lower() for ext in DEFAULT_SUPPORTED_EXTENSIONS
        ]

    def test_accepts_the_five_new_formats(self):
        from rag_parser.frontend.app import ACCEPTED_TYPES

        for extension in ("doc", "odt", "rtf", "xml", "md"):
            assert extension in ACCEPTED_TYPES, f"{extension} отвергается UI"

    def test_streamlit_receives_extensions_without_dots(self):
        """Streamlit ждёт расширения БЕЗ точки, а не MIME-типы."""
        from rag_parser.frontend.app import ACCEPTED_TYPES

        assert all("." not in item for item in ACCEPTED_TYPES)
        assert all("/" not in item for item in ACCEPTED_TYPES)
        assert len(ACCEPTED_TYPES) == len(SUPPORTED_EXTENSIONS)

    def test_unsupported_extensions_absent(self):
        from rag_parser.frontend.app import ACCEPTED_TYPES

        for extension in ("pptx", "exe", "epub", "zip"):
            assert extension not in ACCEPTED_TYPES

    def test_help_text_mentions_every_format(self):
        from rag_parser.frontend.app import ACCEPTED_HELP, ACCEPTED_TYPES

        for extension in ACCEPTED_TYPES:
            assert extension in ACCEPTED_HELP, f"{extension} не упомянут в help"
        assert str(len(ACCEPTED_TYPES)) in ACCEPTED_HELP

    def test_no_hardcoded_list_in_app_source(self):
        """Список не должен дублироваться литералом в коде UI."""
        import inspect

        from rag_parser.frontend import app as frontend

        source = inspect.getsource(frontend)
        assert "DEFAULT_SUPPORTED_EXTENSIONS" in source
        assert 'ACCEPTED_TYPES = ["docx"' not in source

    def test_formats_endpoint_matches_uploader(self):
        from rag_parser.frontend.app import ACCEPTED_TYPES

        api_extensions = [item.lstrip(".").lower() for item in supported_extensions()]
        assert sorted(ACCEPTED_TYPES) == sorted(api_extensions)
