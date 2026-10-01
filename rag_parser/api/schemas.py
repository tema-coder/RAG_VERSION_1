from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChunkingType(str, Enum):
    NONE = "none"
    FIXED = "fixed"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"


class ParseRequest(BaseModel):
    chunking_type: ChunkingType = ChunkingType.NONE
    chunk_size: int = Field(default=500, ge=100, le=2000)
    overlap: int = Field(default=50, ge=0, le=500)
    table_chunk_size: int = Field(default=5, ge=1, le=50)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class ChunkResponse(BaseModel):
    index: int
    text: str
    tokens: int
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    text_for_embedding: Optional[str] = None


class ParseResponse(BaseModel):
    document_id: str
    filename: str
    format: str
    pages: Optional[int] = None
    markdown: str
    chunks: Optional[List[ChunkResponse]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    detail: str
    error: Optional[str] = None
