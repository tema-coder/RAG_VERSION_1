"""Пакетная обработка папки ``input/``.

Скрипт проходит по всем поддерживаемым файлам, прогоняет их через полный
пайплайн ``DedocParser`` -> ``HybridRagChunker`` и складывает результаты в
``output/``:

* ``{stem}_parsed.md``  — итоговый Markdown;
* ``{stem}_chunks.json`` — чанки с ``breadcrumb``, ``parent_id``, ``chunk_order``;
* ``{stem}_tree.json``  — структурное дерево документа.

Настройки те же, что и у API: ``chunk_size``, ``overlap`` и ``threshold``
читаются из ``core.config`` (или ``.env``).

Запуск (из корня репозитория):

    python -m rag_parser.scripts.process_input_folder
    python -m rag_parser.scripts.process_input_folder --input-dir input --output-dir output

Устойчивость: ошибка на одном файле логируется и не прерывает цикл — важно
для пакетной обработки, где один битый файл не должен ронять остальные.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import orjson
import structlog
from tqdm import tqdm

if __package__ in (None, ""):  # прямой запуск python scripts/....py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag_parser.chunkers.hybrid import HybridRagChunker  # noqa: E402
from rag_parser.core.config import settings  # noqa: E402
from rag_parser.core.interfaces import (  # noqa: E402
    Chunk,
    DocumentError,
    ParseResult,
)
from rag_parser.parsers.dedoc_parser import SUPPORTED_EXTENSIONS, DedocParser  # noqa: E402

logger = structlog.get_logger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PACKAGE_ROOT.parent / "input"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "output"

MD_SUFFIX = "_parsed.md"
CHUNKS_SUFFIX = "_chunks.json"
TREE_SUFFIX = "_tree.json"

#: Расширения, которые всегда считаются «нашими», даже если встретятся
#: дважды с разным регистром (``.DOCX`` и ``.docx``).
_SKIPPED_DIR_PREFIXES = ("~$", ".~lock.")


def configure_logging(level: Optional[str] = None) -> None:
    """Настраивает structlog на вывод в stderr."""
    resolved = (level or settings.LOG_LEVEL or "INFO").upper()
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, resolved, logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, resolved, logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def collect_files(input_dir: Path, recursive: bool = True) -> List[Path]:
    """Возвращает файлы с поддерживаемыми расширениями, отсортированные по имени.

    Расширения сравниваются в нижнем регистре (``.DOCX`` == ``.docx``).
    Временные файлы LibreOffice (``~$name.docx``) пропускаются.
    """
    supported = {ext.lower() for ext in SUPPORTED_EXTENSIONS}
    pattern = "**/*" if recursive else "*"
    found = [
        item
        for item in sorted(input_dir.glob(pattern))
        if item.is_file()
        and item.suffix.lower() in supported
        and not item.name.startswith(_SKIPPED_DIR_PREFIXES)
    ]
    return sorted(found, key=lambda item: (item.stem.lower(), item.name.lower()))


def _unique_path(path: Path) -> Path:
    """Не перетирает результат предыдущего файла с тем же stem (``a.docx``/``a.txt``)."""
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    return path


def process_file(
    path: Path,
    output_dir: Path,
    parser: Optional[DedocParser] = None,
    chunker: Optional[HybridRagChunker] = None,
) -> Dict[str, Optional[Path]]:
    """Обрабатывает один файл и пишет три артефакта.

    :raises DocumentError: если разбор не удался (поймает вызывающий).
    """
    parser = parser or DedocParser()
    result: ParseResult = parser.parse(path)

    if chunker is not None:
        result.chunks = chunker.chunk(
            result.markdown,
            {"source": path.name, "format": result.format},
            tree=result.tree,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = _unique_path(output_dir / f"{path.stem}{MD_SUFFIX}")
    chunks_path = _unique_path(output_dir / f"{path.stem}{CHUNKS_SUFFIX}")
    tree_path = _unique_path(output_dir / f"{path.stem}{TREE_SUFFIX}")

    markdown_path.write_bytes(result.markdown.encode("utf-8"))
    chunks_payload = [
        chunk.to_dict() for chunk in (result.chunks or [])
    ]
    chunks_path.write_bytes(
        orjson.dumps(chunks_payload, option=orjson.OPT_INDENT_2) + b"\n"
    )
    tree_payload = result.tree.to_dict() if result.tree is not None else {}
    tree_path.write_bytes(
        orjson.dumps(tree_payload, option=orjson.OPT_INDENT_2) + b"\n"
    )
    return {
        "markdown": markdown_path,
        "chunks": chunks_path,
        "tree": tree_path,
    }


def _build_chunker(
    text_strategy: str,
    chunk_size: Optional[int],
    overlap: Optional[int],
    threshold: Optional[float],
) -> HybridRagChunker:
    return HybridRagChunker(
        text_strategy=text_strategy,  # type: ignore[arg-type]
        chunk_size=chunk_size,
        overlap=overlap,
        threshold=threshold,
    )


def run(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    recursive: bool = True,
    chunking: bool = True,
    text_strategy: str = "token",
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    threshold: Optional[float] = None,
    header_threshold: Optional[float] = None,  # noqa: ARG001 - DEPRECATED
) -> Dict[str, object]:
    """Обрабатывает папку целиком. Ошибка файла не прерывает цикл.

    :param header_threshold: **устаревший**, игнорируется: распознавание
        заголовков детерминированное, порога у него нет.

    :return: ``{"processed", "failed", "total", "artifacts", "errors"}``
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.is_dir():
        logger.warning("input_dir_missing", path=str(input_dir))
        return {"processed": 0, "failed": 0, "total": 0, "artifacts": {}, "errors": []}

    files = collect_files(input_dir, recursive=recursive)
    logger.info(
        "batch_start",
        input=str(input_dir),
        output=str(output_dir),
        total=len(files),
    )
    if not files:
        logger.warning("no_supported_files", input=str(input_dir))
        return {"processed": 0, "failed": 0, "total": 0, "artifacts": {}, "errors": []}

    parser = DedocParser()
    chunker = (
        _build_chunker(text_strategy, chunk_size, overlap, threshold)
        if chunking
        else None
    )

    artifacts: Dict[str, Dict[str, Path]] = {}
    errors: List[Dict[str, str]] = []

    for path in tqdm(files, desc="Обработка", unit="файл"):
        logger.info("processing", filename=path.name)
        try:
            written = process_file(path, output_dir, parser=parser, chunker=chunker)
        except DocumentError as exc:
            # Ошибка разбора одного файла не должна ронять остальные.
            logger.error("parse_failed", filename=path.name, error=str(exc))
            errors.append({"filename": path.name, "error": str(exc)})
            continue
        except Exception as exc:  # noqa: BLE001 - пакетный режим: любая ошибка
            logger.error(
                "parse_failed", filename=path.name, error=str(exc), exc_info=True
            )
            errors.append({"filename": path.name, "error": str(exc)})
            continue
        artifacts[path.name] = written
        logger.info(
            "processed",
            filename=path.name,
            markdown=written["markdown"].name,
            chunks=written["chunks"].name,
            tree=written["tree"].name,
        )

    summary = {
        "processed": len(artifacts),
        "failed": len(errors),
        "total": len(files),
        "artifacts": artifacts,
        "errors": errors,
    }
    logger.info(
        "batch_done",
        processed=summary["processed"],
        failed=summary["failed"],
        total=summary["total"],
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    defaults = argparse.ArgumentParser(
        prog="rag_parser.scripts.process_input_folder",
        description="Пакетный разбор папки input/ в output/ (парсинг + чанкинг)",
    )
    defaults.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    defaults.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    defaults.add_argument(
        "--no-recursive", action="store_true", help="не обходить подкаталоги"
    )
    defaults.add_argument(
        "--no-chunking", action="store_true", help="только парсинг, без чанков"
    )
    defaults.add_argument(
        "--text-strategy",
        default="token",
        choices=("token", "sentence", "semantic", "recursive"),
    )
    defaults.add_argument("--chunk-size", type=int, default=None)
    defaults.add_argument("--overlap", type=int, default=None)
    defaults.add_argument("--threshold", type=float, default=None)
    defaults.add_argument("--log-level", default=None)

    args = defaults.parse_args(argv)
    if args.log_level:
        configure_logging(args.log_level)

    summary = run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        recursive=not args.no_recursive,
        chunking=not args.no_chunking,
        text_strategy=args.text_strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        threshold=args.threshold,
    )

    print(
        f"Processed {summary['processed']}/{summary['total']} files successfully"
    )
    if summary["failed"]:
        print(f"Failed: {summary['failed']}")
        for item in summary["errors"]:
            print(f"  - {item['filename']}: {item['error'][:120]}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI-точка входа
    raise SystemExit(main())


__all__ = [
    "Chunk",
    "DEFAULT_INPUT_DIR",
    "DEFAULT_OUTPUT_DIR",
    "collect_files",
    "main",
    "process_file",
    "run",
]