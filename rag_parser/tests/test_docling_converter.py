"""Тесты адаптера Docling (без загрузки моделей)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from rag_parser.parsers import docling_converter
from rag_parser.parsers.docling_converter import DoclingError


class _FakeDocument:
    def __init__(self, markdown: str) -> None:
        self._markdown = markdown
        self.export_calls = 0

    def export_to_markdown(self) -> str:
        self.export_calls += 1
        return self._markdown


class _FakeConverter:
    def __init__(self, *, markdown: str = "## Title\n\nBody", status=None,
                 errors=None, pages=None) -> None:
        from docling.datamodel.document import ConversionStatus

        self.document = _FakeDocument(markdown)
        self.status = status or ConversionStatus.SUCCESS
        self.errors = errors or []
        self.pages = pages if pages is not None else {0: object()}
        self.paths = []

    def convert(self, path):
        self.paths.append(Path(path))
        return self


@pytest.fixture(autouse=True)
def _reset_docling():
    docling_converter.reset_singletons()
    yield
    docling_converter.reset_singletons()


def _pdf_bytes() -> bytes:
    return b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"


class TestPdfConversion:
    def test_convert_pdf_returns_markdown(self, tmp_path, monkeypatch):
        source = tmp_path / "doc.pdf"
        source.write_bytes(_pdf_bytes())
        fake = _FakeConverter(markdown="# Header\n\nParagraph")
        monkeypatch.setattr(docling_converter, "get_converter", lambda: fake)

        result = docling_converter.convert_to_markdown(source)

        assert result.markdown == "# Header\n\nParagraph"
        assert result.source == "pdf"
        assert result.page_count == 1
        assert fake.paths == [source]

    def test_docling_comments_are_cleaned(self, tmp_path, monkeypatch):
        source = tmp_path / "doc.pdf"
        source.write_bytes(_pdf_bytes())
        fake = _FakeConverter(
            markdown="# H\n\n<!-- image -->\n\nText\n\n<!-- page-break -->\n"
        )
        monkeypatch.setattr(docling_converter, "get_converter", lambda: fake)

        result = docling_converter.convert_to_markdown(source)

        assert "<!--" not in result.markdown
        assert result.markdown == "# H\n\nText"

    def test_failure_status_raises(self, tmp_path, monkeypatch):
        from docling.datamodel.document import ConversionStatus

        source = tmp_path / "doc.pdf"
        source.write_bytes(_pdf_bytes())
        fake = _FakeConverter(
            status=ConversionStatus.FAILURE, errors=["bad pdf"]
        )
        monkeypatch.setattr(docling_converter, "get_converter", lambda: fake)

        with pytest.raises(DoclingError) as excinfo:
            docling_converter.convert_to_markdown(source)
        assert excinfo.value.error_type == "docling_conversion_failure"
        assert "bad pdf" in str(excinfo.value)

    def test_empty_markdown_raises(self, tmp_path, monkeypatch):
        source = tmp_path / "doc.pdf"
        source.write_bytes(_pdf_bytes())
        monkeypatch.setattr(
            docling_converter, "get_converter", lambda: _FakeConverter(markdown="")
        )
        with pytest.raises(DoclingError) as excinfo:
            docling_converter.convert_to_markdown(source)
        assert excinfo.value.error_type == "docling_empty_markdown"

    def test_converter_exception_is_wrapped(self, tmp_path, monkeypatch):
        source = tmp_path / "doc.pdf"
        source.write_bytes(_pdf_bytes())

        class _Boom:
            def convert(self, path):
                raise RuntimeError("boom")

        monkeypatch.setattr(docling_converter, "get_converter", lambda: _Boom())
        with pytest.raises(DoclingError) as excinfo:
            docling_converter.convert_to_markdown(source)
        assert excinfo.value.error_type == "docling_conversion_failure"
        assert "boom" in str(excinfo.value)


class TestImageConversion:
    def _png(self, tmp_path: Path, mode: str = "RGB") -> Path:
        path = tmp_path / "scan.png"
        image = Image.new(mode, (80, 40), "white" if mode != "RGBA" else (255, 255, 255, 255))
        image.save(path)
        return path

    def test_image_is_packed_to_temp_pdf_and_cleaned(
        self, tmp_path, monkeypatch
    ):
        source = self._png(tmp_path)
        original_bytes = source.read_bytes()
        fake = _FakeConverter(markdown="Текст изображения")
        seen: list = []

        def _get_converter():
            seen.extend(fake.paths)
            return fake

        monkeypatch.setattr(docling_converter, "get_converter", _get_converter)

        result = docling_converter.convert_to_markdown(source)

        assert result.source == "image"
        assert result.metadata["source_extension"] == ".png"
        assert len(fake.paths) == 1
        temp_pdf = fake.paths[0]
        assert temp_pdf.suffix == ".pdf"
        assert not temp_pdf.exists(), "временный PDF должен быть удалён"
        assert source.read_bytes() == original_bytes, "оригинал не перезаписан"
        assert temp_pdf != source

    def test_alpha_png_is_supported(self, tmp_path, monkeypatch):
        source = self._png(tmp_path, mode="RGBA")
        fake = _FakeConverter(markdown="Прозрачный PNG")
        monkeypatch.setattr(docling_converter, "get_converter", lambda: fake)

        result = docling_converter.convert_to_markdown(source)

        assert result.markdown == "Прозрачный PNG"
        assert not fake.paths[0].exists()

    def test_temp_pdf_cleaned_on_docling_failure(self, tmp_path, monkeypatch):
        source = self._png(tmp_path)
        paths: list = []

        class _Boom:
            def convert(self, path):
                paths.append(Path(path))
                raise RuntimeError("docling down")

        monkeypatch.setattr(docling_converter, "get_converter", lambda: _Boom())

        with pytest.raises(DoclingError):
            docling_converter.convert_to_markdown(source)
        assert paths and not paths[0].exists()

    def test_img2pdf_failure_is_reported(self, tmp_path, monkeypatch):
        source = self._png(tmp_path)
        import img2pdf

        def _boom(*args, **kwargs):
            raise RuntimeError("bad image")

        monkeypatch.setattr(img2pdf, "convert", _boom)
        monkeypatch.setattr(
            docling_converter, "_normalize_image_for_pdf", lambda path: None
        )
        with pytest.raises(DoclingError) as excinfo:
            docling_converter.convert_to_markdown(source)
        assert excinfo.value.error_type == "image_to_pdf_failure"


class TestInputValidation:
    def test_missing_file(self, tmp_path):
        with pytest.raises(DoclingError) as excinfo:
            docling_converter.convert_to_markdown(tmp_path / "missing.pdf")
        assert excinfo.value.error_type == "file_not_found"

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.pdf"
        path.write_bytes(b"")
        with pytest.raises(DoclingError) as excinfo:
            docling_converter.convert_to_markdown(path)
        assert excinfo.value.error_type == "empty_file"

    def test_unsupported_extension(self, tmp_path):
        path = tmp_path / "file.docx"
        path.write_bytes(b"PK\x03\x04")
        with pytest.raises(DoclingError) as excinfo:
            docling_converter.convert_to_markdown(path)
        assert excinfo.value.error_type == "unsupported_format"
