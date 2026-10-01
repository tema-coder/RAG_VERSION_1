import asyncio
import json
import time

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
from rag_parser.parsers.csv_parser import CsvParser
from rag_parser.parsers.markitdown_parser import MarkItDownParser
from rag_parser.parsers.xlsx_parser import XlsxParser
from rag_parser.core.config import settings
from rag_parser.core.interfaces import (
    DocumentParseError,
    FileTooLargeError,
    UnsupportedFormatError,
)
from rag_parser.main import app
from rag_parser.tests.conftest import _build_pdf, _text_ops

CHUNKING_TYPES = ["none", "fixed", "sentence", "semantic"]


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    document_store = DocumentStore(max_items=4)
    import rag_parser.api.routes as routes

    routes.document_store = document_store
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    routes.document_store = document_store


@pytest.fixture
def all_documents(pdf_files, docx_files, xlsx_files, csv_files, html_files, text_files):
    return {
        "txt": text_files["txt_utf8"],
        "md": text_files["md"],
        "csv": csv_files["csv_semicolon"],
        "html": html_files["html_full"],
        "docx": docx_files["docx_headings"],
        "xlsx": xlsx_files["xlsx_multi"],
        "pdf": pdf_files["pdf_table"],
    }


def upload(path, **data):
    return {"file": (path.name, path.read_bytes(), "application/octet-stream")}, data


def _hub_reachable() -> bool:
    import socket

    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
        return True
    except OSError:
        return False


class TestMetaEndpoints:
    def test_health(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_formats(self, client):
        response = client.get("/api/formats")
        assert response.status_code == 200
        body = response.json()
        assert ".pdf" in body["supported_extensions"]
        assert body["max_file_size_mb"] == settings.MAX_FILE_SIZE_MB
        assert body["chunking_types"] == CHUNKING_TYPES

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["supported_extensions"]

    def test_openapi(self, client):
        assert client.get("/openapi.json").status_code == 200


class TestParse:
    @pytest.mark.parametrize("kind", ["txt", "md", "csv", "html", "docx", "xlsx", "pdf"])
    def test_all_formats(self, client, all_documents, kind):
        path = all_documents[kind]
        files, data = upload(path)
        response = client.post("/api/parse", files=files, data=data)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["filename"] == path.name
        assert body["document_id"]
        assert body["markdown"].strip()
        assert body["chunks"] is None
        assert body["metadata"]

    def test_pdf_markdown_metadata(self, client, pdf_files):
        files, data = upload(pdf_files["pdf_table"])
        response = client.post("/api/parse", files=files, data=data)
        assert response.status_code == 200
        body = response.json()
        assert body["format"] == "pdf"
        assert body["pages"] is None
        assert body["metadata"]["format"] == "pdf"
        assert body["metadata"]["converter"] == "markitdown"
        assert body["metadata"]["size_bytes"] > 0
        assert "Quarterly report" in body["markdown"]
        assert "| Region" in body["markdown"]

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
            assert table_chunks
            for chunk in table_chunks:
                assert chunk["metadata"]["chunker"] == "table"
                assert chunk["text_for_embedding"]
                assert "|" not in chunk["text_for_embedding"]

    def test_table_chunks_carry_text_for_embedding(self, client, pdf_files):
        files, data = upload(
            pdf_files["pdf_table"], chunking_type="fixed", chunk_size="100"
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
        files, data = upload(text_files["md"])
        response = client.post("/api/parse", files=files)
        assert response.status_code == 200
        assert response.json()["chunks"] is None

    def test_missing_file_field(self, client):
        response = client.post("/api/parse", data={"chunking_type": "none"})
        assert response.status_code == 400


class TestParseErrors:
    def test_unsupported_extension(self, client, tmp_path):
        path = tmp_path / "virus.exe"
        path.write_bytes(b"MZ\x90\x00binary")
        files, data = upload(path)
        response = client.post("/api/parse", files=files, data=data)
        assert response.status_code == 400
        assert "Unsupported" in response.json()["detail"]

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
            {"table_chunk_size": "0"},
            {"table_chunk_size": "100"},
        ],
    )
    def test_invalid_chunk_parameters(self, client, text_files, params):
        files, data = upload(text_files["md"], **params)
        response = client.post("/api/parse", files=files, data=data)
        assert response.status_code == 400

    def test_internal_error_is_500(self, client, text_files, monkeypatch):
        import rag_parser.api.routes as routes

        class Boom:
            format = "text"

            def parse(self, file_path):
                raise RuntimeError("boom")

        monkeypatch.setattr(routes, "get_parser", lambda filename: Boom())
        files, data = upload(text_files["md"])
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
        assert "Quarterly report" in response.text

    def test_download_json(self, client, xlsx_files):
        files, data = upload(xlsx_files["xlsx_multi"], chunking_type="fixed", chunk_size="100")
        document_id = client.post("/api/parse", files=files, data=data).json()["document_id"]
        response = client.get(f"/api/parse/{document_id}/download?format=json")
        assert response.status_code == 200
        payload = json.loads(response.content)
        assert payload["document_id"] == document_id
        assert payload["format"] == "xlsx"
        assert payload["chunks"]
        assert payload["chunks"][0]["metadata"]["chunk_type"] in ("text", "table")

    def test_download_unknown_document(self, client):
        assert client.get("/api/parse/does-not-exist/download").status_code == 404

    def test_download_invalid_format(self, client, text_files):
        files, data = upload(text_files["md"])
        document_id = client.post("/api/parse", files=files, data=data).json()["document_id"]
        assert client.get(f"/api/parse/{document_id}/download?format=docx").status_code == 400

    def test_store_is_bounded(self, client, text_files):
        files, data = upload(text_files["md"])
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

        assert body["format"] == "pdf"
        assert "Page 1 of the document." in body["markdown"]
        assert "Page 5 of the document." in body["markdown"]
        for cell in ("Item", "Amount", "Nails", "Boards", "250"):
            assert cell in body["markdown"]
        assert body["chunks"]
        kinds = set()
        for index, chunk in enumerate(body["chunks"]):
            assert chunk["index"] == index
            assert chunk["tokens"] <= 128
            assert chunk["text_for_embedding"] is None
            kinds.add(chunk["metadata"]["chunk_type"])
        assert kinds == {"text"}

        document_id = body["document_id"]
        markdown = client.get(f"/api/parse/{document_id}/download?format=md")
        assert markdown.status_code == 200
        assert markdown.text == body["markdown"]

        payload = client.get(f"/api/parse/{document_id}/download?format=json").json()
        assert payload["markdown"] == body["markdown"]
        assert len(payload["chunks"]) == len(body["chunks"])
        assert payload["metadata"]["converter"] == "markitdown"

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
        started = time.perf_counter()
        response = client.post("/api/parse", files=files, data=data)
        elapsed = time.perf_counter() - started
        assert response.status_code == 200
        assert "Page 10." in response.json()["markdown"]
        assert elapsed < 5.0, f"parsing took {elapsed:.2f}s"


class TestDependenciesUnit:
    def test_get_parser_by_extension(self):
        from rag_parser.chunkers.hybrid import HybridRagChunker

        assert isinstance(get_parser("file.PDF"), MarkItDownParser)
        assert isinstance(get_parser("file.xlsx"), XlsxParser)
        assert isinstance(get_parser("file.xls"), XlsxParser)
        assert isinstance(get_parser("file.docx"), MarkItDownParser)
        assert isinstance(get_parser("file.md"), MarkItDownParser)
        assert isinstance(get_parser("file.csv"), CsvParser)
        assert isinstance(get_parser("file.tsv"), CsvParser)

    def test_get_parser_unsupported(self):
        with pytest.raises(UnsupportedFormatError):
            get_parser("file.exe")

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
        assert ".pdf" in extensions
        assert ".csv" in extensions
        assert ".xlsm" not in extensions
        assert ".pptx" not in extensions

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
    async def test_parse_with_httpx(self, text_files):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as async_client:
            health = await async_client.get("/api/health")
            assert health.status_code == 200

            path = text_files["md"]
            files = {"file": (path.name, path.read_bytes(), "text/markdown")}
            response = await async_client.post(
                "/api/parse", files=files, data={"chunking_type": "sentence", "chunk_size": "100"}
            )
            assert response.status_code == 200
            document_id = response.json()["document_id"]
            download = await async_client.get(
                f"/api/parse/{document_id}/download?format=json"
            )
            assert download.status_code == 200
            assert download.json()["document_id"] == document_id

    async def test_concurrent_requests(self, pdf_files):
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
    def test_process_function(self, pdf_files):
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
        assert result["format"] == "pdf"
        assert result["markdown"]
        assert result["chunks"]
        assert result["document_id"]

    def test_process_raises_document_error(self, invalid_files):
        from rag_parser.frontend.app import process

        class Upload:
            name = "broken.pdf"

            def getbuffer(self):
                return invalid_files["broken_pdf"].read_bytes()

        with pytest.raises(DocumentParseError):
            process(Upload(), ChunkingType.NONE, 500, 50)
