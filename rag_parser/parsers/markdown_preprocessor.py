"""Препроцессинг Markdown (``.md``) → HTML.

Зачем это нужно
---------------
Раньше ``.md`` копировался в ``.txt`` и уходил в ``RawTextReader``, который
не знает синтаксиса Markdown: ``# Заголовок`` становится обычным абзацем,
а pipe-таблица — текстом. Итог: все чанки получали один ``parent_id``,
таблицы не распознавались.

Теперь ``.md`` рендерится в HTML через ``mistune`` (плагин ``table``) и
уходит в ``HtmlReader``, где заголовки дают иерархию, а таблицы проходят
тот же маркерный маршрут, что и обычный HTML (см. ``html_preprocessor``).

Почему ``mistune``, а не ``markdown``
-----------------------------------
``mistune`` — чистый Python без транзитивных зависимостей, поддерживает
GFM-таблицы из коробки и уже зафиксирован в ``requirements.txt``. Пакет
``markdown`` тянет за собой ``pygments``/``mdurl`` и тяжелее.

Ограничения
-----------
* Сырой HTML внутри ``.md`` экранируется (дефолт ``mistune``) — это
  защищает от внедрения произвольной разметки в dedoc.
* Диаграммы/сноски не поддерживаются: это неструктурированный текст.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import mistune

from rag_parser.parsers.html_preprocessor import inject_table_markers

__all__ = ["markdown_to_html", "prepare_markdown_for_dedoc"]

#: Плагины mistune. ``table`` обязателен — без него pipe-таблицы
#: рендерятся как абзацы текста и теряются (ровно тот баг, который чиним).
_MARKDOWN_PLUGINS = ("table", "strikethrough", "url")

_MARKDOWN = mistune.create_markdown(plugins=_MARKDOWN_PLUGINS)

_HTML_WRAPPER_OPEN = "<html><body>"
_HTML_WRAPPER_CLOSE = "</body></html>"

#: Маркер жирного текста, который переживает ``HtmlReader``.
BOLD_MARKER = "__BOLD__"
#: Теги, которые ``HtmlReader`` выбрасывает, оставляя только текст.
_BOLD_TAGS = ("strong", "b")


def _keep_bold(html: str) -> str:
    """Заменяет ``<strong>``/``<b>`` на маркеры, которые видит dedoc.

    ``HtmlReader`` извлекает только текст: ``<strong>Текст</strong>``
    превратился бы в «Текст», и признак «жирный» исчез бы до того, как
    документ дойдёт до ``classify_header``. Поэтому заменяем теги на
    текстовый маркер ``__BOLD__Текст__BOLD__``, который переживает
    разбор; ``strip_bold_markers()`` убирает его перед генерацией
    Markdown.
    """
    from bs4 import BeautifulSoup

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:  # noqa: BLE001 - битый HTML читаем как есть
        return html
    for name in _BOLD_TAGS:
        for tag in soup.find_all(name):
            tag.replace_with(f"{BOLD_MARKER}{tag.get_text()}{BOLD_MARKER}")
    return str(soup)


def markdown_to_html(text: str) -> str:
    """Рендерит Markdown в полноценный HTML-документ."""
    if not text or not text.strip():
        return _HTML_WRAPPER_OPEN + _HTML_WRAPPER_CLOSE
    body = _keep_bold(_MARKDOWN(text))
    return _HTML_WRAPPER_OPEN + body + _HTML_WRAPPER_CLOSE


def _read_text(file_path: Path) -> Optional[str]:
    """Читает файл как UTF-8; при BOM снимает его, иначе ``None``."""
    try:
        raw = file_path.read_bytes()
    except OSError:
        return None
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def prepare_markdown_for_dedoc(
    file_path: Path,
    target_dir: Path,
) -> Tuple[Path, List[str]]:
    """Готовит ``.md`` к разбору dedoc.

    Возвращает ``(путь_к_временному_html, markdown_таблицы)``. Если в файле
    нет таблиц, всё равно возвращается временный HTML — заголовки-то нужны.
    """
    text = _read_text(file_path)
    if text is None:
        # Нечитаемый файл: отдаём как есть, разберётся детектор dedoc.
        return file_path, []

    html, tables = inject_table_markers(markdown_to_html(text))
    target_dir.mkdir(parents=True, exist_ok=True)
    temp_path = target_dir / f"{file_path.stem}_from_md.html"
    temp_path.write_text(html, encoding="utf-8")
    return temp_path, tables