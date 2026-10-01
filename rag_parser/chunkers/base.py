from typing import Dict

import tiktoken

from rag_parser.core.interfaces import BaseChunker, Chunk

ENCODING_NAME = "cl100k_base"
_ENCODINGS: Dict[str, tiktoken.Encoding] = {}


def get_encoding(name: str = ENCODING_NAME) -> tiktoken.Encoding:
    if name not in _ENCODINGS:
        _ENCODINGS[name] = tiktoken.get_encoding(name)
    return _ENCODINGS[name]


def count_tokens(text: str, encoding_name: str = ENCODING_NAME) -> int:
    if not text:
        return 0
    return len(get_encoding(encoding_name).encode(text, disallowed_special=()))


__all__ = ["BaseChunker", "Chunk", "ENCODING_NAME", "count_tokens", "get_encoding"]
