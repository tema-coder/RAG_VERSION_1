"""DOCX parsing through the `dedoc` library.

dedoc builds a tree of nodes out of the document (headers keep their hierarchy
level, tables become separate structures) which is far richer than a plain
"convert to text" pipeline. Two artifacts are produced per document and stored
on disk under ``output/dedoc_results/``:

* ``raw_dedoc_tree_{document_id}.json`` - the raw dedoc document tree;
* ``parsed_markdown_{document_id}.md``     - the Markdown rendering of that tree.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import orjson

from rag_parser.core.interfaces import BaseParser, DocumentParseError, ParseResult

SUPPORTED_EXTENSIONS: List[str] = [".docx"]

TREE_FILE_PREFIX = "raw_dedoc_tree"
MARKDOWN_FILE_PREFIX = "parsed_markdown"

_manager: Any = None


def _build_manager():
    """Assemble a dedoc pipeline tuned for structured DOCX input.

    Only the DOCX reader is wired up, and the default structure extractor plus
    the tree constructor are used, so header levels and tables survive the
    conversion. The default manager config would additionally instantiate every
    reader and converter dedoc ships with, most of which are irrelevant here.
    """
    from dedoc import DedocManager
    from dedoc.api.md_out_type import json2md
    from dedoc.attachments_handler import AttachmentsHandler
    from dedoc.config import get_config
    from dedoc.converters import ConverterComposition
    from dedoc.converters.concrete_converters.docx_converter import DocxConverter
    from dedoc.metadata_extractors import MetadataExtractorComposition
    from dedoc.metadata_extractors.concrete_metadata_extractors import (
        base_metadata_extractor,
        docx_metadata_extractor,
    )
    from dedoc.readers import DocxReader, ReaderComposition
    from dedoc.structure_constructors import StructureConstructorComposition
    from dedoc.structure_constructors.concrete_structure_constructors import (
        tree_constructor,
    )
    from dedoc.structure_extractors import StructureExtractorComposition
    from dedoc.structure_extractors.concrete_structure_extractors import (
        default_structure_extractor,
    )

    config = get_config()
    structure_extractor_cls = default_structure_extractor.DefaultStructureExtractor
    tree_constructor_instance = tree_constructor.TreeConstructor()
    manager = DedocManager(
        config=config,
        manager_config={
            "converter": ConverterComposition(
                converters=[DocxConverter(config=config)]
            ),
            "reader": ReaderComposition(readers=[DocxReader(config=config)]),
            "structure_extractor": StructureExtractorComposition(
                extractors={
                    structure_extractor_cls.document_type: structure_extractor_cls(
                        config=config
                    )
                },
                default_key=structure_extractor_cls.document_type,
                config=config,
            ),
            "structure_constructor": StructureConstructorComposition(
                constructors={
                    "tree": tree_constructor_instance,
                    "": tree_constructor_instance,
                },
                default_constructor=tree_constructor_instance,
            ),
            "document_metadata_extractor": MetadataExtractorComposition(
                extractors=[
                    docx_metadata_extractor.DocxMetadataExtractor(config=config),
                    base_metadata_extractor.BaseMetadataExtractor(config=config),
                ]
            ),
            "attachments_handler": AttachmentsHandler(config=config),
        },
    )
    return manager, json2md


def get_manager() -> Tuple[Any, Any]:
    """Returns the cached ``(DedocManager, json2md)`` pair.

    Building the pipeline imports half of dedoc, so it is created once and
    reused across requests.
    """
    global _manager
    if _manager is None:
        _manager = _build_manager()
    return _manager


def dedoc_version() -> str:
    try:
        import dedoc.version

        return dedoc.version.__version__
    except Exception:  # noqa: BLE001 - version info is optional metadata
        return "unknown"


class DedocParser(BaseParser):
    """Structured DOCX parser: dedoc tree on disk + Markdown in the response."""

    format = "dedoc"
    extensions = (".docx",)

    def supported_extensions(self) -> List[str]:
        return list(SUPPORTED_EXTENSIONS)

    def parse(self, file_path: Path) -> ParseResult:
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
                f"Unsupported extension {extension!r} for Dedoc parser. "
                f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
                filename=path.name,
            )

        try:
            manager, json2md = get_manager()
        except ImportError as exc:
            raise DocumentParseError(
                f"dedoc is not installed: {exc}. See requirements-dedoc.txt",
                filename=path.name,
            ) from exc

        try:
            parsed = manager.parse(
                str(path),
                parameters={"with_attachments": "false", "structure_type": "tree"},
            )
            document = parsed.to_api_schema()
            tree = document.model_dump()
            markdown = json2md(document)
        except DocumentParseError:
            raise
        except Exception as exc:  # noqa: BLE001 - dedoc raises assorted errors
            raise DocumentParseError(
                f"dedoc failed to parse the document: {exc}", filename=path.name
            ) from exc

        if not markdown.strip():
            raise DocumentParseError(
                "Converted document contains no text", filename=path.name
            )
        markdown = self._clean(markdown)

        document_id = self._new_document_id()
        paths = self.save_results(document_id, tree, markdown)
        tables = tree.get("content", {}).get("tables") or []
        return ParseResult(
            document_id=document_id,
            filename=path.name,
            format=self.format,
            markdown=markdown,
            metadata={
                "format": self.format,
                "extension": extension,
                "size_bytes": size_bytes,
                "lines": len(markdown.splitlines()),
                "characters": len(markdown),
                "parser": "dedoc",
                "dedoc_version": dedoc_version(),
                "tables": len(tables),
                "raw_tree_path": str(paths["raw_tree"]),
                "markdown_path": str(paths["markdown"]),
                "output_dir": str(paths["output_dir"]),
            },
        )

    def _clean(self, text: str) -> str:
        """dedoc separates block elements with up to four newlines."""
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n\n")
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def save_results(
        self, document_id: str, tree: Dict[str, Any], markdown: str
    ) -> Dict[str, Path]:
        """Writes the raw dedoc tree and the rendered Markdown to disk."""
        from rag_parser.core.config import settings

        output_dir = settings.ensure_dedoc_output_dir()
        tree_path = output_dir / f"{TREE_FILE_PREFIX}_{document_id}.json"
        markdown_path = output_dir / f"{MARKDOWN_FILE_PREFIX}_{document_id}.md"
        tree_path.write_bytes(
            orjson.dumps(tree, option=orjson.OPT_INDENT_2) + b"\n"
        )
        markdown_path.write_bytes(markdown.encode("utf-8"))
        return {
            "output_dir": output_dir,
            "raw_tree": tree_path,
            "markdown": markdown_path,
        }