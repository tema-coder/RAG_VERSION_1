import logging
import sys
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from rag_parser.api.dependencies import supported_extensions
from rag_parser.api.routes import router
from rag_parser.core.config import settings

API_VERSION = "1.0.0"


def configure_logging(level: Optional[str] = None) -> None:
    log_level = (level or settings.LOG_LEVEL or "INFO").upper()
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level, logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level, logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


configure_logging()
logger = structlog.get_logger(__name__)

app = FastAPI(
    title="Document Parser with Chunking",
    version=API_VERSION,
    description=(
        "Сервис парсинга документов различных форматов в Markdown "
        "с опциональным чанкингом."
    ),
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(
        "request",
        method=request.method,
        path=request.url.path,
    )
    response = await call_next(request)
    logger.info(
        "response",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    logger.warning("validation_error", path=request.url.path, errors=exc.errors())
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid request parameters", "error": "invalid_params"},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_error", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {exc}", "error": "internal_error"},
    )


app.include_router(router)


@app.get("/", tags=["meta"])
async def root() -> dict:
    return {
        "name": "Document Parser with Chunking",
        "version": API_VERSION,
        "docs": "/docs",
        "supported_extensions": supported_extensions(),
    }
