"""Тесты Child-Parent логики (ТЗ 3.4).

``expand_to_parents`` — чистая функция, проверяется на моковых чанках без
API и без document_store. Интеграционные тесты идут через реальный
POST /api/parse и GET /api/parse/{id}/parents.
"""

import pytest

from rag_parser.core.interfaces import Chunk
from rag_parser.core.retrieval import (
    chunk_from_dict,
    chunks_from_dicts,
    expand_to_parents,
    group_by_parent,
)


def _chunk(index, parent_id=None, order=0, kind="text", text=None):
    return Chunk(
        index=index,
        text=text if text is not None else f"чанк {index}",
        tokens=5,
        source="doc.docx",
        metadata={"chunk_type": kind},
        parent_id=parent_id,
        chunk_order=order,
    )


def mixed_chunks_fixture():
    """Вступление-сирота + два раздела (текст и таблица) + раздел 3."""
    return [
        _chunk(0, None, 0, "text", "Вступление"),
        _chunk(1, "h_1", 0, "text", "Раздел 1"),
        _chunk(2, "h_1", 1, "text", "Абзац раздела 1"),
        _chunk(3, "h_2", 0, "text", "Раздел 2"),
        _chunk(4, "h_2", 1, "text", "Абзац раздела 2"),
        _chunk(5, "h_2", 2, "table", "| A | B |"),
        _chunk(6, "h_2", 3, "table", "| 1 | 2 |"),
        _chunk(7, "h_3", 0, "text", "Раздел 3"),
    ]


@pytest.fixture
def mixed_chunks():
    return mixed_chunks_fixture()


class TestExpandUnit:
    def test_expand_orphan_returns_single(self):
        """Нет parent_id → возвращается ровно один чанк (ТЗ 3.4)."""
        chunks = [_chunk(0, None), _chunk(1, None)]
        result = expand_to_parents(chunks, 1)
        assert len(result) == 1
        assert result[0] is chunks[1]

    def test_expand_group_returns_sorted_by_order(self):
        """Группа родителя отдаётся по возрастанию chunk_order."""
        chunks = [
            _chunk(0, "h_1", 2, "text", "третий"),
            _chunk(1, "h_1", 0, "text", "первый"),
            _chunk(2, "h_1", 1, "text", "второй"),
        ]
        result = expand_to_parents(chunks, 0)
        assert [c.chunk_order for c in result] == [0, 1, 2]
        assert [c.text for c in result] == ["первый", "второй", "третий"]

    def test_expand_preserves_table_and_text_chunks_in_group(self):
        """Текстовые и табличные чанки одного раздела в одной группе."""
        chunks = mixed_chunks_fixture()
        result = expand_to_parents(chunks, 5)
        assert len(result) == 4  # Раздел 2: заголовок, абзац, 2 строки таблицы
        kinds = [c.metadata["chunk_type"] for c in result]
        assert kinds == ["text", "text", "table", "table"]
        assert [c.chunk_order for c in result] == [0, 1, 2, 3]

    def test_expand_from_any_position_of_group(self):
        """Любой чанк группы разворачивает её целиком."""
        chunks = mixed_chunks_fixture()
        from_first = [c.index for c in expand_to_parents(chunks, 3)]
        from_last = [c.index for c in expand_to_parents(chunks, 6)]
        assert from_first == from_last == [3, 4, 5, 6]

    def test_expand_does_not_mutate_input(self):
        chunks = mixed_chunks_fixture()
        before = [c.index for c in chunks]
        expand_to_parents(chunks, 5)
        assert [c.index for c in chunks] == before

    def test_expand_empty_chunks(self):
        with pytest.raises(IndexError):
            expand_to_parents([], 0)

    def test_expand_index_out_of_range(self):
        chunks = mixed_chunks_fixture()
        with pytest.raises(IndexError):
            expand_to_parents(chunks, 99)
        with pytest.raises(IndexError):
            expand_to_parents(chunks, -1)

    def test_group_by_parent(self):
        groups = group_by_parent(mixed_chunks_fixture())
        assert set(groups) == {"h_1", "h_2", "h_3"}
        assert [c.chunk_order for c in groups["h_2"]] == [0, 1, 2, 3]


class TestDictRoundTrip:
    def test_chunk_from_dict_roundtrip(self):
        original = _chunk(4, "h_2", 2, "table", "| A | B |")
        restored = chunk_from_dict(original.to_dict())
        assert restored == original

    def test_chunk_from_dict_defaults(self):
        restored = chunk_from_dict({})
        assert restored.parent_id is None
        assert restored.chunk_order == 0
        assert restored.index == 0

    def test_chunk_from_dict_ignores_unknown_keys(self):
        restored = chunk_from_dict({"index": 1, "что-то": "лишнее"})
        assert restored.index == 1

    def test_chunks_from_dicts(self):
        chunks = mixed_chunks_fixture()
        restored = chunks_from_dicts([c.to_dict() for c in chunks])
        assert restored == chunks


class TestApiIntegration:
    def _parse(self, client, path, chunking="fixed"):
        with open(path, "rb") as handle:
            files = {"file": (path.name, handle.read(), "application/octet-stream")}
        response = client.post(
            "/api/parse", files=files, data={"chunking_type": chunking}
        )
        assert response.status_code == 200, response.text
        return response.json()

    def test_endpoint_returns_whole_parent_group(self, client, docx_files):
        body = self._parse(client, docx_files["docx_headings"])
        document_id = body["document_id"]
        chunks = body["chunks"] or []
        assert chunks

        # Находим чанк с непустым parent_id.
        target = next(c for c in chunks if c["parent_id"])
        expected_group = [
            c for c in chunks if c["parent_id"] == target["parent_id"]
        ]
        assert len(expected_group) >= 2, "нужен раздел из нескольких чанков"

        response = client.get(
            f"/api/parse/{document_id}/parents", params={"index": target["index"]}
        )
        assert response.status_code == 200
        group = response.json()
        assert len(group) == len(expected_group)
        assert [c["chunk_order"] for c in group] == sorted(
            c["chunk_order"] for c in group
        )
        assert [c["index"] for c in group] == [c["index"] for c in expected_group]
        assert all(c["parent_id"] == target["parent_id"] for c in group)

    def test_endpoint_for_orphan_returns_single_chunk(self, client, xlsx_files):
        """Документ без заголовков: у всех чанков parent_id = None."""
        body = self._parse(client, xlsx_files["xlsx_single"])
        chunks = body["chunks"] or []
        assert chunks
        orphan = next(c for c in chunks if c["parent_id"] is None)

        response = client.get(
            f"/api/parse/{body['document_id']}/parents",
            params={"index": orphan["index"]},
        )
        assert response.status_code == 200
        group = response.json()
        assert len(group) == 1
        assert group[0]["index"] == orphan["index"]

    def test_endpoint_error_cases(self, client, docx_files):
        body = self._parse(client, docx_files["docx_headings"])
        document_id = body["document_id"]

        missing = client.get("/api/parse/nonexistent-id/parents", params={"index": 0})
        assert missing.status_code == 404

        out_of_range = client.get(
            f"/api/parse/{document_id}/parents", params={"index": 99999}
        )
        assert out_of_range.status_code == 400
        assert out_of_range.headers.get("X-Error-Type") == "index_out_of_range"

    def test_negative_index_rejected(self, client, docx_files):
        """index < 0 отсекается валидатором (в приложении все ошибки
        валидации приводятся к 400 глобальным обработчиком)."""
        body = self._parse(client, docx_files["docx_headings"])
        response = client.get(
            f"/api/parse/{body['document_id']}/parents", params={"index": -1}
        )
        assert response.status_code == 400