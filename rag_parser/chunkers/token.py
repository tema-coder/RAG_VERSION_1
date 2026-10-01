from typing import List, Optional

from chonkie import TokenChunker

from rag_parser.chunkers.base import BaseChunker, Chunk

CHUNKER_NAME = "token"


class TokenChunkerRAG(BaseChunker):
    """Fixed-size token chunker backed by chonkie ``TokenChunker``."""

    name = CHUNKER_NAME

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        source: str = "document",
    ) -> None:
        from rag_parser.core.config import settings

        self.chunk_size = int(
            settings.DEFAULT_CHUNK_SIZE if chunk_size is None else chunk_size
        )
        self.overlap = int(
            settings.DEFAULT_OVERLAP if overlap is None else overlap
        )
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        self.source = source
        self.chunker = TokenChunker(
            tokenizer="cl100k_base",
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
        )

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        metadata = metadata or {}
        if not text or not text.strip():
            return []
        source = metadata.get("source", self.source)
        extra = {key: value for key, value in metadata.items() if key != "source"}
        chunks: List[Chunk] = []
        for index, raw in enumerate(self.chunker.chunk(text)):
            chunks.append(
                Chunk(
                    index=index,
                    text=raw.text,
                    tokens=raw.token_count,
                    source=source,
                    metadata={
                        "chunk_index": index,
                        "chunker": self.name,
                        "token_count": raw.token_count,
                        "chunk_size": self.chunk_size,
                        "overlap": self.overlap,
                        **extra,
                    },
                )
            )
        return chunks


# Backwards-compatible alias for the pre-chonkie implementation.
FixedTokenChunker = TokenChunkerRAG
