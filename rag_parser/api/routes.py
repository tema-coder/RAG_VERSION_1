import logging
import time
from typing import Optional
from urllib.parse import quote

import orjson
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response

from rag_parser.api.dependencies import (
    document_store,
    get_chunker,
    get_parser,
    save_upload,
    supported_extensions,
    validate_upload,
)
from rag_parser.api.schemas import (
    ChunkingType,
    ChunkResponse,
    ErrorResponse,
    HealthResponse,
    ParseResponse,
)
from rag_parser.core.config import settings
from rag_parser.core.interfaces import (
    DocumentError,
    DocumentParseError,
    FileTooLargeError,
    ParseResult,
    UnsupportedFormatError,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["documents"])


def _error(status_code: int, message: str, error: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=message,
        headers={"X-Error-Type": error},
    )


def _result_to_response(result: ParseResult) -> ParseResponse:
    chunks = None
    if result.chunks is not None:
        chunks = [
            ChunkResponse(
                index=chunk.index,
                text=chunk.text,
                tokens=chunk.tokens,
                source=chunk.source,
                metadata=chunk.metadata,
                text_for_embedding=chunk.text_for_embedding,
            )
            for chunk in result.chunks
        ]
    metadata = dict(result.metadata)
    return ParseResponse(
        document_id=result.document_id,
        filename=result.filename,
        format=result.format,
        pages=metadata.get("pages"),
        markdown=result.markdown,
        chunks=chunks,
        metadata=metadata,
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/formats")
async def formats() -> dict:
    return {
        "supported_extensions": supported_extensions(),
        "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
        "chunking_types": [item.value for item in ChunkingType],
    }


@router.post(
    "/parse",
    response_model=ParseResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def parse_document(
    file: UploadFile = File(...),
    chunking_type: ChunkingType = Form(ChunkingType.NONE),
    chunk_size: Optional[int] = Form(None),
    overlap: Optional[int] = Form(None),
    table_chunk_size: Optional[int] = Form(None),
    threshold: Optional[float] = Form(None),
) -> ParseResponse:
    started = time.perf_counter()
    filename = file.filename or ""

    if chunk_size is None:
        chunk_size = settings.DEFAULT_CHUNK_SIZE
    if overlap is None:
        overlap = settings.DEFAULT_OVERLAP
    if table_chunk_size is None:
        table_chunk_size = settings.DEFAULT_TABLE_CHUNK_SIZE
    if threshold is None:
        threshold = settings.SEMANTIC_THRESHOLD
    if overlap >= chunk_size:
        raise _error(400, "overlap must be smaller than chunk_size", "invalid_params")
    if chunk_size < 100 or chunk_size > 2000:
        raise _error(
            400, "chunk_size must be between 100 and 2000", "invalid_params"
        )
    if overlap < 0 or overlap > 500:
        raise _error(400, "overlap must be between 0 and 500", "invalid_params")
    if table_chunk_size < 1 or table_chunk_size > 50:
        raise _error(
            400, "table_chunk_size must be between 1 and 50", "invalid_params"
        )
    if not 0.0 < threshold < 1.0:
        raise _error(
            400, "threshold must be strictly between 0 and 1", "invalid_params"
        )

    data = await file.read()
    if len(data) > settings.max_file_size_bytes:
        raise _error(
            413,
            f"File exceeds limit of {settings.MAX_FILE_SIZE_MB} MB",
            "file_too_large",
        )

    try:
        validate_upload(filename, data)
        parser = get_parser(filename)
    except UnsupportedFormatError as exc:
        raise _error(400, exc.message, "unsupported_format") from exc
    except FileTooLargeError as exc:
        raise _error(413, exc.message, "file_too_large") from exc
    except DocumentError as exc:
        raise _error(400, exc.message, "invalid_file") from exc

    temp_path = save_upload(filename, data)
    try:
        result = parser.parse(temp_path)
        result.filename = filename
        chunker = get_chunker(
            ChunkingType(chunking_type),
            chunk_size,
            overlap,
            source=filename,
            threshold=threshold,
            table_chunk_size=table_chunk_size,
        )
        if chunker is not None:
            result.chunks = chunker.chunk(
                result.markdown, {"source": filename, "format": result.format}
            )
    except UnsupportedFormatError as exc:
        raise _error(400, exc.message, "unsupported_format") from exc
    except DocumentParseError as exc:
        raise _error(400, exc.message, "parse_error") from exc
    except ValueError as exc:
        raise _error(400, str(exc), "invalid_params") from exc
    except Exception as exc:  # noqa: BLE001 - last resort guard
        logger.exception("Failed to parse %s", filename)
        raise _error(500, f"Parsing failed: {exc}", "internal_error") from exc
    finally:
        temp_path.unlink(missing_ok=True)

    response = _result_to_response(result)
    document_store.add(
        {
            "document_id": result.document_id,
            "filename": result.filename,
            "format": result.format,
            "markdown": result.markdown,
            "metadata": result.metadata,
            "chunks": [chunk.to_dict() for chunk in result.chunks or []],
        }
    )
    logger.info(
        "parsed %s format=%s chunks=%d chars=%d elapsed=%.3fs",
        filename,
        result.format,
        len(result.chunks or []),
        len(result.markdown),
        time.perf_counter() - started,
    )
    return response


@router.get(
    "/parse/{document_id}/download",
    responses={200: {"content": {"application/octet-stream": {}}}, 404: {}},
)
async def download_document(
    document_id: str,
    format: str = Query("md", pattern="^(md|json)$"),
) -> Response:
    document = document_store.get(document_id)
    if document is None:
        raise _error(404, f"Document {document_id} not found", "not_found")

    if format == "json":
        payload = orjson.dumps(
            {
                "document_id": document["document_id"],
                "filename": document["filename"],
                "format": document["format"],
                "markdown": document["markdown"],
                "metadata": document["metadata"],
                "chunks": document["chunks"],
            },
            option=orjson.OPT_INDENT_2,
        )
        return Response(
            content=payload,
            media_type="application/json",
            headers={
                "Content-Disposition": (
                    f"attachment; filename*=UTF-8''{quote(document['filename'])}.json"
                )
            },
        )

    return Response(
        content=document["markdown"],
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": (
                f"attachment; filename*=UTF-8''{quote(document['filename'])}.md"
            )
        },
    )
