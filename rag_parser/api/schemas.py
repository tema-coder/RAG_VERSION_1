from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChunkingType(str, Enum):
    NONE = "none"
    FIXED = "fixed"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"


class ParseRequest(BaseModel):
    chunking_type: ChunkingType = ChunkingType.NONE
    chunk_size: int = Field(default=500, ge=100, le=2000)
    overlap: int = Field(default=50, ge=0, le=500)
    threshold: float = Field(default=0.5, gt=0.0, lt=1.0)


class ChunkResponse(BaseModel):
    index: int
    text: str
    tokens: int
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    text_for_embedding: Optional[str] = None
    breadcrumb: Optional[str] = None
    parent_id: Optional[str] = None
    chunk_order: int = 0


class TreeNode(BaseModel):
    type: str
    level: int
    text: str
    children: List["TreeNode"] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


TreeNode.model_rebuild()


class ParseResponse(BaseModel):
    document_id: str
    filename: str
    format: str
    pages: Optional[int] = None
    markdown: str
    chunks: Optional[List[ChunkResponse]] = None
    tree: Optional[TreeNode] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    detail: str
    error: Optional[str] = None
