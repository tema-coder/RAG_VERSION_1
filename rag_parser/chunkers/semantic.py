from typing import List, Optional

from rag_parser.chunkers.base import BaseChunker, Chunk

CHUNKER_NAME = "semantic"
DEFAULT_THRESHOLD = 0.5
DEFAULT_MODEL_NAME = "deepvk/USER-bge-m3"

_SENTENCE_END_RE = None


def _count_sentences(text: str) -> int:
    import re

    global _SENTENCE_END_RE
    if _SENTENCE_END_RE is None:
        _SENTENCE_END_RE = re.compile(r"[.!?]+\s+")
    stripped = (text or "").strip()
    if not stripped:
        return 0
    return max(1, len(_SENTENCE_END_RE.split(stripped)))


def resolve_embedding_id(model: Optional[str] = None) -> str:
    """Builds a chonkie embedding identifier for a sentence-transformers model."""
    from rag_parser.core.config import settings

    name = model or settings.SEMANTIC_MODEL_NAME or DEFAULT_MODEL_NAME
    if "://" in name:
        return name
    return f"sentence-transformers/{name}"


class SemanticChunkerRAG(BaseChunker):
    """Semantic chunker backed by chonkie ``SemanticChunker``.

    The embedding model (default ``deepvk/USER-bge-m3`` via
    ``settings.SEMANTIC_MODEL_NAME``) is downloaded from the Hugging Face Hub
    on first use (~2 GB, CPU-only). The chonkie chunker is therefore created
    lazily inside :meth:`chunk`, so importing or instantiating this class
    never requires network access.
    """

    name = CHUNKER_NAME

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        source: str = "document",
        threshold: Optional[float] = None,
        model: Optional[str] = None,
        min_sentences_per_chunk: int = 2,
    ) -> None:
        from rag_parser.core.config import settings

        self.chunk_size = int(
            settings.DEFAULT_CHUNK_SIZE if chunk_size is None else chunk_size
        )
        self.overlap = int(settings.DEFAULT_OVERLAP if overlap is None else overlap)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self.threshold = float(
            settings.SEMANTIC_THRESHOLD if threshold is None else threshold
        )
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be strictly between 0 and 1")
        self.model = model or settings.SEMANTIC_MODEL_NAME or DEFAULT_MODEL_NAME
        self.source = source
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self._chunker = None

    def _get_chunker(self):
        if self._chunker is None:
            from chonkie import SemanticChunker as _ChonkieSemanticChunker

            self._chunker = _ChonkieSemanticChunker(
                embedding_model=resolve_embedding_id(self.model),
                threshold=self.threshold,
                chunk_size=self.chunk_size,
                min_sentences_per_chunk=self.min_sentences_per_chunk,
            )
        return self._chunker

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        metadata = metadata or {}
        if not text or not text.strip():
            return []
        source = metadata.get("source", self.source)
        extra = {key: value for key, value in metadata.items() if key != "source"}
        chunks: List[Chunk] = []
        for index, raw in enumerate(self._get_chunker().chunk(text)):
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
                        "sentences": _count_sentences(raw.text),
                        "semantic_score": None,
                        "threshold": self.threshold,
                        "encoder": "sentence-transformers",
                        "model": self.model,
                        "chunk_size": self.chunk_size,
                        "overlap": self.overlap,
                        "min_sentences_per_chunk": self.min_sentences_per_chunk,
                        **extra,
                    },
                )
            )
        return chunks


# Backwards-compatible alias for the pre-chonkie implementation.
SemanticChunker = SemanticChunkerRAG
