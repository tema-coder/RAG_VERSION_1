"""Диагностика стилей абзацев DOCX (Шаг 3 плана диагностики task.txt).

Для каждого абзаца выводит: индекс, имя стиля Word, кегль первого run,
флаг bold и превью текста. Помогает ответить на вопрос Гипотезы 3:
распознал ли Word абзац «Раздел N» как Heading, или это обычный текст
(и тогда классификация ложится целиком на ``header_promoter``).

Запуск::

    python -m rag_parser.scripts.check_docx_styles path/to/file.docx
    python -m rag_parser.scripts.check_docx_styles            # по умолчанию input/2023.docx

Скрипт НЕ зависит от dedoc — только от ``python-docx``, поэтому работает
даже в окружении без тяжёлых зависимостей парсера.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


def _run_size(para) -> Optional[float]:
    """Кегль первого run с заданным размером (иначе ``None``)."""
    for run in para.runs:
        if run.font.size is not None:
            return run.font.size.pt
    return None


def _bold_flag(para) -> str:
    """``B`` — весь абзац жирный, ``b`` — часть, ``-`` — ничего."""
    flags = [bool(run.font.bold) for run in para.runs]
    if not flags:
        return "-"
    if all(flags):
        return "B"
    if any(flags):
        return "b"
    return "-"


def check_styles(path: Path, limit: int = 80) -> None:
    """Печатает стили первых ``limit`` непустых абзацев ``path``."""
    import docx  # python-docx

    doc = docx.Document(str(path))
    print(f"=== {path.name}: всего абзацев {len(doc.paragraphs)} ===")
    print(f"=== СТИЛИ АБЗАЦЕВ (первые {limit} непустых) ===")
    print(f"{'#':>4} {'style':16} {'sz':>6} {'bold':>4}  text")
    print("-" * 90)
    shown = 0
    for index, para in enumerate(doc.paragraphs):
        text = (para.text or "").strip()
        if not text:
            continue
        style = para.style.name if para.style else "None"
        size = _run_size(para)
        size_str = f"{size:.1f}" if size else "-"
        bold = _bold_flag(para)
        print(f"{index:>4} {style[:16]:16} {size_str:>6} {bold:>4}  {text[:70]}")
        shown += 1
        if shown >= limit:
            break

    # Целевой поиск «Раздел N» — ключевая строка для диагностики 2023.docx.
    print("\n=== АБЗАЦЫ «Раздел N» (стиль + жирность) ===")
    for index, para in enumerate(doc.paragraphs):
        text = (para.text or "").strip()
        if (
            text.lower().startswith("раздел ")
            and text[:12].replace("раздел ", "").rstrip(".").split()[:1]
            and any(ch.isdigit() for ch in text[:14])
        ):
            style = para.style.name if para.style else "None"
            bold = _bold_flag(para)
            size = _run_size(para)
            size_str = f"{size:.1f}" if size else "-"
            print(
                f"#{index:>3} [{style[:14]:14}] sz={size_str:>5} {bold} | {text[:75]}"
            )


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    path = Path(argv[0]) if argv else Path("input/2023.docx")
    if not path.exists():
        # откат на загруженный эталон, если дефолтного пути нет
        fallback = Path("/home/z/my-project/upload/2023.docx")
        if fallback.exists():
            path = fallback
        else:
            print(f"File not found: {path}", file=sys.stderr)
            return 2
    limit = int(argv[1]) if len(argv) > 1 else 80
    check_styles(path, limit=limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
