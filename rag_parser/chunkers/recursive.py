from typing import List, Optional, Sequence

from chonkie import RecursiveChunker as _ChonkieRecursiveChunker
from chonkie.types.recursive import RecursiveLevel, RecursiveRules

from rag_parser.chunkers.base import BaseChunker, Chunk, get_encoding

CHUNKER_NAME = "recursive"
TOKENIZER_NAME = "cl100k_base"

DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP = 20
DEFAULT_SEPARATORS: List[str] = ["\n\n", "\n", ". ", " "]
DEFAULT_MIN_CHARACTERS_PER_CHUNK = 1


def build_rules(separators: Sequence[str]) -> RecursiveRules:
    """Translate a separator hierarchy into chonkie ``RecursiveRules``.

    Separators are applied top-down, so their order is the splitting priority:
    paragraphs first, then lines, then sentences, then words. Whitespace-only
    separators cannot be passed as delimiters (chonkie rejects them), they are
    mapped to a dedicated whitespace level instead.
    """
    levels: List[RecursiveLevel] = []
    delimiters: List[str] = []
    for separator in separators:
        if separator.strip():
            delimiters.append(separator)
            continue
        if delimiters:
            levels.append(RecursiveLevel(delimiters=delimiters, include_delim="prev"))
            delimiters = []
        levels.append(RecursiveLevel(whitespace=True, include_delim="prev"))
    if delimiters:
        levels.append(RecursiveLevel(delimiters=delimiters, include_delim="prev"))
    return RecursiveRules(levels=levels)


class RecursiveChunker(BaseChunker):
    """Hierarchy-aware chunker backed by chonkie ``RecursiveChunker``.

    chonkie splits the text recursively: the first separator level that keeps
    a chunk inside ``chunk_size`` wins, so paragraphs and sentences stay intact
    whenever possible.

    ``chunk_overlap`` is not supported by chonkie's recursive chunker, so the
    tail of every previous chunk is prepended here to keep the window size
    configurable and consistent with the other chunkers.
    """

    name = CHUNKER_NAME

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        separators: Optional[Sequence[str]] = None,
        source: str = "document",
        min_characters_per_chunk: int = DEFAULT_MIN_CHARACTERS_PER_CHUNK,
    ) -> None:
        self.chunk_size = int(
            DEFAULT_CHUNK_SIZE if chunk_size is None else chunk_size
        )
        self.overlap = int(DEFAULT_OVERLAP if overlap is None else overlap)
        self.separators = list(
            DEFAULT_SEPARATORS if separators is None else separators
        )
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        if not self.separators:
            raise ValueError("separators must not be empty")
        self.source = source
        self.min_characters_per_chunk = min_characters_per_chunk
        self.chunker = _ChonkieRecursiveChunker(
            # Позиционно: 1.3.x — ``tokenizer_or_token_counter``, 1.7.x — ``tokenizer``.
            TOKENIZER_NAME,
            chunk_size=self.chunk_size,
            rules=build_rules(self.separators),
            min_characters_per_chunk=self.min_characters_per_chunk,
        )

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        metadata = metadata or {}
        if not text or not text.strip():
            return []
        source = metadata.get("source", self.source)
        extra = {key: value for key, value in metadata.items() if key != "source"}
        encoding = get_encoding()
        chunks: List[Chunk] = []
        previous: Optional[str] = None

        for index, raw in enumerate(self.chunker.chunk(text)):
            chunk_text = raw.text
            if previous is not None and self.overlap:
                chunk_text = self._tail(previous, encoding) + chunk_text
            previous = raw.text
            chunks.append(
                Chunk(
                    index=index,
                    text=chunk_text,
                    tokens=len(encoding.encode(chunk_text, disallowed_special=())),
                    source=source,
                    metadata={
                        "chunk_index": index,
                        "chunker": self.name,
                        "token_count": raw.token_count,
                        "chunk_size": self.chunk_size,
                        "overlap": self.overlap,
                        "separators": list(self.separators),
                        "start_index": raw.start_index,
                        "end_index": raw.end_index,
                        **extra,
                    },
                )
            )
        return chunks

    def _tail(self, text: str, encoding) -> str:
        """Last ``overlap`` tokens of ``text``, decoded back to a string."""
        token_ids = encoding.encode(text, disallowed_special=())
        return encoding.decode(token_ids[-self.overlap :])


__all__ = ["RecursiveChunker", "build_rules"]