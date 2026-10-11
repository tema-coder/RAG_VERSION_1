"""Тесты пакетной обработки папки input/.

Все тесты работают на tmp_path и не трогают реальные input/ и output/.
"""

import orjson
import pytest

from rag_parser.core.config import settings
from rag_parser.scripts import process_input_folder as script


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Изолированные input/ и output/ + настройки парсера на tmp_path."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    monkeypatch.setattr(
        settings, "DEDOC_OUTPUT_DIR", str(tmp_path / "dedoc_results")
    )
    monkeypatch.setattr(settings, "TEMP_DIR", str(tmp_path / "temp"))
    return input_dir, output_dir


def _make_docx(path):
    from docx import Document

    document = Document()
    document.add_heading("Раздел 1. Общие сведения", level=1)
    document.add_paragraph("Первый абзац документа.")
    document.add_heading("1.1 Подраздел", level=2)
    document.add_paragraph("Абзац подраздела.")
    document.save(path)
    return path


def _make_txt(path, text="Строка документа.\nВторая строка.\n"):
    path.write_text(text, encoding="utf-8")
    return path


def _artifacts(output_dir):
    return sorted(item.name for item in output_dir.iterdir())


class TestCollectFiles:
    def test_collects_supported_only(self, workspace):
        input_dir, _ = workspace
        _make_docx(input_dir / "a.docx")
        _make_txt(input_dir / "b.txt")
        (input_dir / "c.exe").write_bytes(b"MZ\x90\x00")
        (input_dir / "d.epub").write_bytes(b"PK\x03\x04")

        found = [path.name for path in script.collect_files(input_dir)]
        assert found == ["a.docx", "b.txt"]

    def test_uppercase_extension(self, workspace):
        input_dir, _ = workspace
        _make_txt(input_dir / "UPPER.TXT")
        assert [p.name for p in script.collect_files(input_dir)] == ["UPPER.TXT"]

    def test_skips_libreoffice_temp_files(self, workspace):
        input_dir, _ = workspace
        _make_txt(input_dir / "~$draft.docx", "мусор")
        _make_txt(input_dir / "real.txt")
        found = [p.name for p in script.collect_files(input_dir)]
        assert found == ["real.txt"]

    def test_recursive_flag(self, workspace):
        input_dir, _ = workspace
        nested = input_dir / "sub"
        nested.mkdir()
        _make_txt(nested / "deep.txt")
        _make_txt(input_dir / "top.txt")

        assert len(script.collect_files(input_dir, recursive=True)) == 2
        assert [p.name for p in script.collect_files(input_dir, recursive=False)] == [
            "top.txt"
        ]


class TestRunHappyPath:
    def test_script_processes_valid_files(self, workspace):
        input_dir, output_dir = workspace
        _make_docx(input_dir / "a.docx")
        _make_txt(input_dir / "b.txt")

        summary = script.run(input_dir, output_dir, chunking=True)

        assert summary["processed"] == 2
        assert summary["failed"] == 0
        assert summary["total"] == 2
        # 2 файла × 3 артефакта = 6
        assert len(_artifacts(output_dir)) == 6
        assert _artifacts(output_dir) == [
            "a_chunks.json", "a_parsed.md", "a_tree.json",
            "b_chunks.json", "b_parsed.md", "b_tree.json",
        ]

    def test_markdown_content_is_written(self, workspace):
        input_dir, output_dir = workspace
        _make_txt(input_dir / "note.txt", "Привет, документ.\nВторая строка.\n")

        script.run(input_dir, output_dir, chunking=True)

        markdown = (output_dir / "note_parsed.md").read_text(encoding="utf-8")
        assert "Привет, документ." in markdown
        assert markdown.strip()

    def test_chunks_json_has_parent_fields(self, workspace):
        input_dir, output_dir = workspace
        _make_docx(input_dir / "a.docx")

        script.run(input_dir, output_dir, chunking=True)

        chunks = orjson.loads((output_dir / "a_chunks.json").read_bytes())
        assert chunks
        for chunk in chunks:
            assert "parent_id" in chunk
            assert "chunk_order" in chunk
            assert "breadcrumb" in chunk

    def test_tree_json_is_document_root(self, workspace):
        input_dir, output_dir = workspace
        _make_docx(input_dir / "a.docx")

        script.run(input_dir, output_dir, chunking=True)

        tree = orjson.loads((output_dir / "a_tree.json").read_bytes())
        assert tree["type"] == "document"
        assert isinstance(tree["children"], list)

    def test_without_chunking_writes_empty_chunks(self, workspace):
        input_dir, output_dir = workspace
        _make_txt(input_dir / "a.txt")

        summary = script.run(input_dir, output_dir, chunking=False)

        assert summary["processed"] == 1
        assert orjson.loads((output_dir / "a_chunks.json").read_bytes()) == []


class TestErrorHandling:
    def test_script_handles_broken_file(self, workspace):
        """Битый файл логирует ошибку и НЕ мешает обработке остальных."""
        input_dir, output_dir = workspace
        _make_txt(input_dir / "good.txt")
        broken = input_dir / "broken.docx"
        broken.write_bytes(b"not a zip container")

        summary = script.run(input_dir, output_dir, chunking=True)

        assert summary["processed"] == 1
        assert summary["failed"] == 1
        assert summary["total"] == 2
        assert summary["errors"][0]["filename"] == "broken.docx"
        # Хороший файл всё равно обработан
        assert (output_dir / "good_parsed.md").exists()
        assert not (output_dir / "broken_parsed.md").exists()

    def test_broken_file_does_not_crash_process(self, workspace):
        input_dir, output_dir = workspace
        (input_dir / "x.docx").write_bytes(b"garbage not a docx")
        _make_txt(input_dir / "y.txt")

        # Исключение наружу не пробрасывается — это главная проверка.
        summary = script.run(input_dir, output_dir, chunking=True)
        assert summary["failed"] == 1
        assert summary["processed"] == 1

    def test_empty_folder(self, workspace):
        input_dir, output_dir = workspace
        summary = script.run(input_dir, output_dir)
        assert summary == {"processed": 0, "failed": 0, "total": 0,
                            "artifacts": {}, "errors": []}

    def test_missing_input_dir(self, workspace, tmp_path):
        _, output_dir = workspace
        summary = script.run(tmp_path / "нет-такой-папки", output_dir)
        assert summary["total"] == 0
        assert summary["processed"] == 0


class TestUnsupported:
    def test_script_skips_unsupported_extensions(self, workspace):
        input_dir, output_dir = workspace
        _make_txt(input_dir / "keep.txt")
        (input_dir / "app.exe").write_bytes(b"MZ\x90\x00")
        (input_dir / "book.epub").write_bytes(b"PK\x03\x04")
        (input_dir / "deck.pptx").write_bytes(b"PK\x03\x04")

        summary = script.run(input_dir, output_dir, chunking=True)

        assert summary["processed"] == 1
        assert summary["failed"] == 0
        assert summary["total"] == 1
        assert _artifacts(output_dir) == [
            "keep_chunks.json", "keep_parsed.md", "keep_tree.json"
        ]


class TestNameCollisions:
    def test_same_stem_different_extension(self, workspace):
        """a.docx и a.txt не затирают результаты друг друга."""
        input_dir, output_dir = workspace
        _make_docx(input_dir / "a.docx")
        _make_txt(input_dir / "a.txt")

        summary = script.run(input_dir, output_dir, chunking=True)

        assert summary["processed"] == 2
        assert len(_artifacts(output_dir)) == 6
        assert "a_parsed.md" in _artifacts(output_dir)
        assert "a_parsed_2.md" in _artifacts(output_dir)


class TestCli:
    def test_main_returns_zero(self, workspace, capsys):
        input_dir, output_dir = workspace
        _make_txt(input_dir / "a.txt")

        code = script.main(
            [
                "--input-dir", str(input_dir),
                "--output-dir", str(output_dir),
                "--no-recursive",
            ]
        )

        assert code == 0
        captured = capsys.readouterr()
        assert "Processed 1/1 files successfully" in captured.out
        assert (output_dir / "a_parsed.md").exists()