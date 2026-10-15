import json
import sys
import tempfile
from pathlib import Path

import streamlit as st

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag_parser.api.dependencies import get_chunker, get_parser
from rag_parser.api.schemas import ChunkingType
from rag_parser.core.config import DEFAULT_SUPPORTED_EXTENSIONS, settings
from rag_parser.core.interfaces import DocumentError

CHUNKING_LABELS = {
    "Без чанкинга": ChunkingType.NONE,
    "Фиксированный по токенам": ChunkingType.FIXED,
    "По предложениям": ChunkingType.SENTENCE,
    "Семантический": ChunkingType.SEMANTIC,
    "Рекурсивный": ChunkingType.RECURSIVE,
}

#: Расширения для ``st.file_uploader(type=...)``.
#:
#: Streamlit ждёт именно РАСШИРЕНИЯ без точки (не MIME-типы): в type=
#: подставляется то, что показывается в диалоге выбора файлов и в
#: подсказке загрузчика.
#:
#: Список берётся из ``core.config.DEFAULT_SUPPORTED_EXTENSIONS``, а не
#: дублируется вручную. Хардкод раньше содержал 9 форматов и разошёлся с
#: бэкендом: UI отбрасывал .doc/.odt/.rtf/.xml/.md, хотя парсер их читает.
#: Теперь источник истины один — SUPPORTED_EXTENSIONS.
ACCEPTED_TYPES = [ext.lstrip(".").lower() for ext in DEFAULT_SUPPORTED_EXTENSIONS]

#: Подсказка под загрузчиком. Собирается из того же списка, чтобы текст и
#: фильтр всегда совпадали.
_FORMATS_HELP = ", ".join(
    ext.lstrip(".").lower() for ext in DEFAULT_SUPPORTED_EXTENSIONS
)
ACCEPTED_HELP = (
    f"Поддерживается {len(ACCEPTED_TYPES)} форматов: {_FORMATS_HELP}. "
    "PDF — только текстовый слой, сканы будут отклонены (422, OCR отключён). "
    "Для doc/odt/rtf требуется системный LibreOffice."
)


def process(
    upload,
    chunking_type: ChunkingType,
    chunk_size: int,
    overlap: int,
    threshold: float = 0.5,
) -> dict:
    from rag_parser.chunkers.hybrid import HybridRagChunker
    from rag_parser.parsers.dedoc_tree import truncate

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
        )
        if chunker is not None:
            if isinstance(chunker, HybridRagChunker) and result.tree is not None:
                chunks = chunker.chunk(
                    result.markdown,
                    {"source": upload.name, "format": result.format},
                    tree=result.tree,
                )
            else:
                chunks = chunker.chunk(
                    result.markdown, {"source": upload.name, "format": result.format}
                )
        preview_tree = (
            truncate(result.tree).to_dict() if result.tree is not None else None
        )
        return {
            "document_id": result.document_id,
            "filename": upload.name,
            "format": result.format,
            "markdown": result.markdown,
            "metadata": result.metadata,
            "chunks": chunks,
            "tree": preview_tree,
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


def render_tree_node(node: dict, depth: int = 0) -> None:
    """Рендерит узел дерева через отступы вместо вложенных экспандеров.

    Это предотвращает StreamlitAPIException: Expanders may not be nested.
    """
    node_type = node.get("type", "?")
    level = node.get("level", 0)
    text = node.get("text", "")
    children = node.get("children", [])

    # Визуальный отступ для имитации иерархии
    indent = "  " * depth

    # Иконки для быстрой визуальной навигации
    icons = {
        "header": "📌",
        "table": "📊",
        "paragraph": "📝",
        "list_item": "🔹",
        "raw_text": "📄",
        "document": "📁",
    }
    icon = icons.get(node_type, "⚪")

    label = f"{indent}{icon} {node_type} · level {level} · {text}"

    # Используем monospace для чёткого выравнивания отступов
    st.markdown(f"`{label}`")

    # Рекурсивно рендерим детей с увеличенным отступом
    for child in children:
        render_tree_node(child, depth + 1)


def main() -> None:
    st.set_page_config(page_title="Парсер документов с чанкингом", layout="wide")
    st.title("Парсер документов с чанкингом (dedoc)")

    uploaded = st.file_uploader(
        "Загрузите документ",
        type=ACCEPTED_TYPES,
        help=ACCEPTED_HELP,
    )

    chunking_label = st.selectbox("Тип чанкинга", list(CHUNKING_LABELS.keys()))
    chunk_size = st.slider("chunk_size", 100, 2000, settings.DEFAULT_CHUNK_SIZE, 50)
    overlap = st.slider("overlap", 0, 500, settings.DEFAULT_OVERLAP, 10)
    threshold = st.slider(
        "threshold (semantic)", 0.05, 0.95, settings.SEMANTIC_THRESHOLD, 0.05
    )
    # Слайдеров размера табличного чанка и порога заголовков нет намеренно:
    # табличный чанк всегда содержит ровно одну строку данных (ТЗ 3.3), а
    # распознавание заголовков детерминированное (7 жёстких правил) и
    # настраивается правкой parsers/header_promoter.py, а не вслепую.

    if overlap >= chunk_size:
        st.error("overlap должен быть меньше chunk_size")
        return

    parse_clicked = st.button(
        "Обработать документ", type="primary", disabled=uploaded is None
    )

    if parse_clicked:
        try:
            with st.spinner("Обработка документа..."):
                st.session_state["result"] = process(
                    uploaded,
                    CHUNKING_LABELS[chunking_label],
                    chunk_size,
                    overlap,
                    threshold,
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
    tabs = st.tabs(["Markdown", "Чанки", "Дерево документа", "Метаданные", "Скачать"])

    with tabs[0]:
        st.markdown(result["markdown"])

    with tabs[1]:
        if not chunks:
            st.info("Чанкинг не применялся")
        else:
            st.write(f"Всего чанков: {len(chunks)}")
            for chunk in chunks:
                crumb = f" · [{chunk.breadcrumb}]" if chunk.breadcrumb else ""
                with st.expander(
                    f"Чанк #{chunk.index} · {chunk.tokens} токенов{crumb}"
                ):
                    st.text(chunk.text)
                    if chunk.text_for_embedding:
                        st.caption("Представление для эмбеддинга:")
                        st.text(chunk.text_for_embedding)
                    st.json(chunk.metadata)

    with tabs[2]:
        tree = result.get("tree")
        if not tree:
            st.info("Дерево пустое")
        else:
            for child in tree.get("children", []):
                render_tree_node(child)
            with st.expander("Сырой JSON дерева"):
                st.json(tree)

    with tabs[3]:
        render_metadata(result["metadata"])

    with tabs[4]:
        base_name = Path(result["filename"]).stem
        st.download_button(
            "Скачать Markdown",
            data=result["markdown"],
            file_name=f"{base_name}.md",
            mime="text/markdown",
        )
        st.download_button(
            "Скачать chunks.json",
            data=json.dumps(
                [chunk.to_dict() for chunk in chunks or []],
                ensure_ascii=False,
                indent=2,
            ),
            file_name=f"{base_name}.chunks.json",
            mime="application/json",
        )
        st.download_button(
            "Скачать tree.json",
            data=json.dumps(
                result.get("tree") or {},
                ensure_ascii=False,
                indent=2,
            ),
            file_name=f"{base_name}.tree.json",
            mime="application/json",
        )


main()
