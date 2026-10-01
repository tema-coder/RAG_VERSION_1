import json
import sys
import tempfile
from pathlib import Path

import streamlit as st

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag_parser.api.dependencies import get_chunker, get_parser
from rag_parser.api.schemas import ChunkingType
from rag_parser.core.config import settings
from rag_parser.core.interfaces import DocumentError

CHUNKING_LABELS = {
    "Без чанкинга": ChunkingType.NONE,
    "Фиксированный по токенам": ChunkingType.FIXED,
    "По предложениям": ChunkingType.SENTENCE,
    "Семантический": ChunkingType.SEMANTIC,
}

ACCEPTED_TYPES = ["pdf", "xlsx", "xls", "csv", "docx", "html", "txt", "md"]


def process(
    upload,
    chunking_type: ChunkingType,
    chunk_size: int,
    overlap: int,
    threshold: float = 0.5,
    table_chunk_size: int = 5,
) -> dict:
    suffix = Path(upload.name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(upload.getbuffer())
        temp_path = Path(handle.name)
    try:
        result = get_parser(upload.name).parse(temp_path)
        result.filename = upload.name
        chunks = None
        chunker = get_chunker(
            chunking_type,
            chunk_size,
            overlap,
            source=upload.name,
            threshold=threshold,
            table_chunk_size=table_chunk_size,
        )
        if chunker is not None:
            chunks = chunker.chunk(
                result.markdown, {"source": upload.name, "format": result.format}
            )
        return {
            "document_id": result.document_id,
            "filename": upload.name,
            "format": result.format,
            "markdown": result.markdown,
            "metadata": result.metadata,
            "chunks": chunks,
        }
    finally:
        temp_path.unlink(missing_ok=True)


def render_metadata(metadata: dict) -> None:
    if not metadata:
        return
    columns = st.columns(4)
    order = list(metadata.items())
    for index, (key, value) in enumerate(order):
        with columns[index % 4]:
            if isinstance(value, (dict, list)):
                st.metric(key, f"{type(value).__name__}({len(value)})")
            else:
                st.metric(key, value)


def main() -> None:
    st.set_page_config(page_title="Парсер документов с чанкингом", layout="wide")
    st.title("Парсер документов с чанкингом")

    uploaded = st.file_uploader(
        "Загрузите документ",
        type=ACCEPTED_TYPES,
        help="Поддерживаются: pdf, xlsx, xls, csv, docx, html, txt, md",
    )

    chunking_label = st.selectbox("Тип чанкинга", list(CHUNKING_LABELS.keys()))
    chunk_size = st.slider("chunk_size", 100, 2000, settings.DEFAULT_CHUNK_SIZE, 50)
    overlap = st.slider("overlap", 0, 500, settings.DEFAULT_OVERLAP, 10)
    threshold = st.slider(
        "threshold (semantic)", 0.05, 0.95, settings.SEMANTIC_THRESHOLD, 0.05
    )
    table_chunk_size = st.slider(
        "table_chunk_size", 1, 50, settings.DEFAULT_TABLE_CHUNK_SIZE, 1
    )

    if overlap >= chunk_size:
        st.error("overlap должен быть меньше chunk_size")
        return

    if st.button("Обработать документ", type="primary", disabled=uploaded is None):
        try:
            with st.spinner("Обработка документа..."):
                st.session_state["result"] = process(
                    uploaded,
                    CHUNKING_LABELS[chunking_label],
                    chunk_size,
                    overlap,
                    threshold,
                    table_chunk_size,
                )
        except DocumentError as exc:
            st.error(f"Ошибка обработки: {exc.message}")
            st.session_state.pop("result", None)
        except Exception as exc:  # noqa: BLE001 - UI boundary
            st.error(f"Непредвиденная ошибка: {exc}")
            st.session_state.pop("result", None)

    result = st.session_state.get("result")
    if not result:
        st.info("Загрузите файл и нажмите «Обработать документ»")
        return

    chunks = result["chunks"]
    tabs = st.tabs(["Markdown", "Чанки", "Метаданные", "Скачать"])

    with tabs[0]:
        st.markdown(result["markdown"])

    with tabs[1]:
        if not chunks:
            st.info("Чанкинг не применялся")
        else:
            st.write(f"Всего чанков: {len(chunks)}")
            for chunk in chunks:
                with st.expander(
                    f"Чанк #{chunk.index} · {chunk.tokens} токенов"
                ):
                    st.text(chunk.text)
                    if chunk.text_for_embedding:
                        st.caption("Представление для эмбеддинга:")
                        st.text(chunk.text_for_embedding)
                    st.json(chunk.metadata)

    with tabs[2]:
        render_metadata(result["metadata"])

    with tabs[3]:
        base_name = Path(result["filename"]).stem
        st.download_button(
            "Скачать Markdown",
            data=result["markdown"],
            file_name=f"{base_name}.md",
            mime="text/markdown",
        )
        st.download_button(
            "Скачать JSON",
            data=json.dumps(
                {
                    "document_id": result["document_id"],
                    "filename": result["filename"],
                    "format": result["format"],
                    "markdown": result["markdown"],
                    "metadata": result["metadata"],
                    "chunks": [chunk.to_dict() for chunk in chunks or []],
                },
                ensure_ascii=False,
                indent=2,
            ),
            file_name=f"{base_name}.json",
            mime="application/json",
        )


main()
