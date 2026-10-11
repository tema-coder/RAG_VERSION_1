"""Тесты защиты bbox PDF (parsers/pdf_bbox_guard).

dedoc 2.8 роняет разбор страниц, у которых координаты текста выходят
за границы рендера (``cv2.inRange`` на пустом срезе). Защита обрезает bbox
и не должна ломать ни обычный разбор, ни повторный вызов.
"""

from pathlib import Path

import pytest

from rag_parser.parsers.dedoc_parser import DedocParser, prepare_manager_for_format
from rag_parser.parsers.pdf_bbox_guard import install_bbox_guard
from rag_parser.tests.conftest import _build_pdf

np = pytest.importorskip("numpy")

INPUT_DIR = Path(__file__).resolve().parent.parent.parent / "input"


def _extractor_cls():
    from dedoc.readers.pdf_reader.pdf_image_reader.line_metadata_extractor import (
        metadata_extractor,
    )

    return metadata_extractor.LineMetadataExtractor


class TestInstall:
    def test_install_is_idempotent(self):
        assert install_bbox_guard() is True
        assert install_bbox_guard() is True

        extractor = _extractor_cls()
        first = getattr(
            extractor, f"_{extractor.__name__}__get_color_annotation"
        )
        install_bbox_guard()
        assert (
            getattr(extractor, f"_{extractor.__name__}__get_color_annotation")
            is first
        )

    def test_patch_not_applied_to_non_pdf_manager(self):
        # Подготовка к не-PDF формату не должна ничего ломать.
        assert prepare_manager_for_format(".docx") is not None


class TestClamping:
    def _call(self, box, image):
        """Вызывает пропатченный метод напрямую."""
        from dedocutils.data_structures.bbox import BBox
        from dedoc.readers.pdf_reader.pdf_image_reader.line_metadata_extractor import (
            metadata_extractor,
        )

        from rag_parser.parsers.pdf_bbox_guard import _ClampedTextWithBBox

        install_bbox_guard()
        extractor = metadata_extractor.LineMetadataExtractor
        method = getattr(extractor, f"_{extractor.__name__}__get_color_annotation")
        payload = _ClampedTextWithBBox(bbox=box, text="Отвечают")
        return method(object.__new__(extractor), payload, image)

    def _box(self, x, y, w, h):
        from dedocutils.data_structures.bbox import BBox

        return BBox(x, y, w, h)

    def test_bbox_inside_image_is_unchanged(self):
        image = np.full((100, 200, 3), 255, dtype=np.uint8)
        annotation = self._call(self._box(10, 20, 50, 15), image)
        assert annotation.start == 0
        assert annotation.end == len("Отвечают")

    def test_bbox_outside_image_does_not_raise(self):
        # Регрессия: x за правой границей -> пустой срез -> падение cv2.
        image = np.full((1653, 2339, 3), 255, dtype=np.uint8)
        annotation = self._call(self._box(2493, 647, 204, 23), image)
        assert annotation.red == 0 and annotation.green == 0 and annotation.blue == 0

    def test_bbox_partially_outside_is_clamped(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        # Прямоугольник наполовину за правым краем: обрезка обязана выжить.
        annotation = self._call(self._box(50, 10, 100, 20), image)
        assert annotation.start == 0

    def test_zero_area_bbox_returns_neutral_color(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        annotation = self._call(self._box(10, 10, 0, 0), image)
        assert (annotation.red, annotation.green, annotation.blue) == (0, 0, 0)


class TestRealFiles:
    """Реальные PDF из input/, на которых dedoc падал."""

    @pytest.mark.parametrize(
        "name", ["пдф_таблица.pdf", "gost_report.pdf"]
    )
    def test_known_troublemaker_parses(self, tmp_path, monkeypatch, name):
        from rag_parser.core.config import settings

        source = INPUT_DIR / name
        if not source.exists():
            pytest.skip(f"{name} отсутствует в input/")

        monkeypatch.setattr(
            settings, "DEDOC_OUTPUT_DIR", str(tmp_path / "output")
        )
        result = DedocParser().parse(source)
        assert result.markdown.strip()
        assert "cv2.error" not in result.markdown