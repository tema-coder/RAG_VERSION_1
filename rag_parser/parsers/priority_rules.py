"""Приоритетная система распознавания заголовков с поглощением.

Заменяет девять независимых правил ``header_promoter.classify_header`` на
приоритетную систему: специфичное правило проверяется раньше общего и
«поглощает» строку — первое совпадение завершает проверку. При сомнении
строка остаётся абзацем: лучше пропустить заголовок, чем выдумать.

Спецификация — ``promt_for_right_work_with_head_page.txt``. Реализация —
четыре прохода:

1. **Классификация строки** (``classify_line``): S → P0 → P1 → P2 → P3 →
   P4 → P5. Если сработало S — строка абзац, дальше не проверяется.
2. **Валидация нумерации** (``validate_chains``): для P3 с emphasis=None
   и P2_LETTER строятся цепочки подряд идущих кандидатов; одиночные
   отбрасываются (``demoted_by_chain``).
3. **Ранги** (``assign_ranks``): фиксированный ранг 1 для P0_APP/P1;
   реестр видов ``base(kind)`` для остальных.
4. **Дерево** (``build_tree``): один проход со стеком, родитель = вершина
   после снятия ≥ r.

Область применения: txt, docx без стилей, PDF-origin Markdown. Для
``MARKUP_AUTHORITATIVE_EXTENSIONS`` (.html/.htm/.md/.xml) разметка
источника авторитетна — приоритетная система не применяется.

Признаки форматирования по источникам (проверено по коду):
- dedoc-дерево (docx): аннотации ``bold``, ``size``, ``alignment``,
  ``indentation``, ``spacing``. ``italic`` НЕ отдаётся.
- markdown_preprocessor (md/html/xml): маркер ``__BOLD__`` в тексте.
- text_preprocessor (.txt): маркер ``**`` в тексте.
- Docling-MD (PDF, будущий): ``#``-уровень (``md_level``).

``italic`` не реализован — ни один источник не отдаёт (зачитано как «вне v1»).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Константы и регексы (спецификация промпта, синтаксис проверен тестами)
# --------------------------------------------------------------------------- #

UP = "A-ZА-ЯЁ"
START = rf"[{UP}«\"“(]"  # название начинается с заглавной/кавычки

#: Арабская нумерация: 1 | 1.1 | 1.1.1 (годы и даты не проходят — {1,2} на сегмент)
ARAB = r"\d{1,2}(?:\.\d{1,2}){0,5}"

#: Римская нумерация (strict, с lookahead чтобы не ловить «I» в «IBM»).
ROMAN = (
    r"(?=[IVXLCDM])"
    r"M{0,3}(?:CM|CD|D?C{0,3})"
    r"(?:XC|XL|L?X{0,3})"
    r"(?:IX|IV|V?I{0,3})"
)

# Структурные слова (регистр НЕ игнорировать — только «Раздел» или «РАЗДЕЛ»).
_WORDS = ["Раздел", "Подраздел", "Глава", "Часть", "Том", "Книга", "Параграф"]
W = "|".join(f"{w.capitalize()}|{w.upper()}" for w in _WORDS)

# --- P0: структурные слова + нумерация ---
P0_WORD = re.compile(
    rf"^(?P<w>{W})\s+(?P<n>{ARAB}|[IVXLCDM]+|[{UP}])"
    rf"(?P<sep>[.:]?)(?:\s+(?P<t>{START}.*))?$"
)
P0_APP = re.compile(
    rf"^(?:Приложение|ПРИЛОЖЕНИЕ)\s+(?P<n>[{UP}]|\d{{1,2}})"
    rf"(?:\s*\((?i:обязательное|рекомендуемое|справочное|информационное)\))?"
    rf"(?P<sep>[.:]?)(?:\s+(?P<t>{START}.*))?$"
)
# «Примечание N. Название» — только форма с точкой после числа.
P0_NOTE = re.compile(
    rf"^(?:Примечание|ПРИМЕЧАНИЕ)\s+(?P<n>\d{{1,3}})\.\s+(?P<t>{START}.*)$"
)

# --- P1: секционные слова (строгое совпадение всей строки) ---
P1_RU = re.compile(
    r"^(?i:введение|заключение|аннотация|предисловие|выводы|литература|"
    r"библиография|список\s+литературы|список\s+использованных\s+источников|"
    r"содержание|оглавление|приложение)\s*[:.]?$"
)
P1_EN = re.compile(
    r"^(?i:abstract|introduction|background|related\s+work|literature\s+review|"
    r"methods?|methodology|materials\s+and\s+methods|"
    r"results(?:\s+and\s+discussion)?|discussion|conclusions?|"
    r"acknowledg(?:e)?ments|references|bibliography|appendix)\s*[:.]?$"
)

# --- NUM-паттерны для P2/P3/P5 ---
NUM_ARAB = re.compile(
    rf"^(?P<n>{ARAB})(?P<sep>[.)]|\s)\s*(?P<t>{START}.*)$"
)
NUM_ROMAN = re.compile(
    rf"^(?P<n>{ROMAN})[.)]\s+(?P<t>{START}.*)$"  # разделитель обязателен
)
NUM_LETTER = re.compile(
    rf"^(?P<n>[{UP}])[.)]\s+(?P<t>{START}.*)$"  # t не должен начинаться с инициала
)

# --- S: подавления ---
# S_TOC: маркер страницы в конце («… 1», «... 4», табуляция/пробелы + число).
# ВАЖНО: требуем число после многоточия/точек, иначе «договор...» (троеточие
# в конце предложения) ложно подавляется.
S_TOC = re.compile(r"(?:…|\.{2,}|\t| {3,})\s*\d{1,4}\s*$")
# S_CAPTION: «Таблица / Рисунок / Схема N…», «Table / Figure / Fig. N…»
S_CAPTION = re.compile(
    r"^(?i:таблица|рисунок|рис|схема|table|figure|fig\.?)\s+\d",
)
# S_NOTE: «Примечание / ПРИМЕЧАНИЕ» + необяз. число + тире/двоеточие/конец.
# Важно: P0_NOTE (Примечание N. Название) проверяется РАНЬШЕ в P0, поэтому
# S_NOTE ловит только формы «Примечание — текст», «Примечание: текст»,
# «Примечание 1 — текст» (без точки после числа) — они не заголовки.
S_NOTE = re.compile(
    r"^(?:Примечание|ПРИМЕЧАНИЕ)(?:\s+\d{1,3})?\s*[:—–-]\s*\S"
)
# S_LIST: маркеры списков.
S_LIST = re.compile(r"^(?:[-•—]\s+|[а-яёА-ЯЁ]\)\s+|\d+\)\s+)")

# --- Вспомогательные ---
# Дата: «12.05.2023», «12.05.2023 г.»
_DATE_RE = re.compile(r"\d{1,2}\.\d{1,2}\.(?:\d{2}|\d{4})")
# «Число + единица»: «3.5 мм», «12 кг», «100 шт.»
_UNIT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:мм|см|дм|м|км|г|кг|т|шт|л|мл|с|мин|ч|°C|руб|€|\$|пт|кгц|мгц|ггц)\b", re.IGNORECASE)
# Инициалы «А. С. Пушкин» — буква + точка + пробел + (буква + точка + пробел) +
INITIALS_RE = re.compile(rf"^[{UP}]\.\s*(?:[{UP}]\.\s*)+[А-ЯA-ZЁ]")


def is_caps(s: str) -> bool:
    """≥3 букв, все буквы заглавные. Пробелы/пунктуация игнорируются."""
    letters = [c for c in s if c.isalpha()]
    return len(letters) >= 3 and all(c.isupper() for c in letters)


def _ends_with_terminal_punct(s: str) -> bool:
    """Оканчивается на ``. ; , : ! ?`` (терминальная пунктуация)."""
    s = s.rstrip()
    return bool(s) and s[-1] in ".;,:!?"


def _is_date_or_unit(s: str) -> bool:
    """Дата («12.05.2023») или «число + единица» («3.5 мм»)."""
    return bool(_DATE_RE.search(s) or _UNIT_RE.search(s))


# --------------------------------------------------------------------------- #
# emphasis — три состояния (True / False / None)
# --------------------------------------------------------------------------- #

def emphasis_from_annotations(
    annotations: List[Dict[str, Any]],
    text: str,
    base_size: Optional[float],
) -> Optional[bool]:
    """Определяет emphasis по аннотациям dedoc (docx-источник).

    :return: ``True`` если жирный ≥90% строки ИЛИ кегль ≥13pt и > базового;
             ``False`` если аннотации есть, но строка не выделена;
             ``None`` если аннотаций нет (источник не даёт форматирование).
    """
    if not annotations:
        return None
    length = len(text)
    if length <= 0:
        return None

    # Bold: покрытие ≥90%.
    bold_covered = 0
    has_bold_ann = False
    for item in annotations:
        if str(item.get("name", "")).lower() != "bold":
            continue
        has_bold_ann = True
        try:
            start = int(item.get("start", 0) or 0)
            end = int(item.get("end", 0) or 0)
        except (TypeError, ValueError):
            continue
        bold_covered = max(bold_covered, end - start)
        if _is_truthy(item.get("value")) and (end - start) / length >= 0.9:
            return True

    # Size: кегль ≥13pt и > базового.
    if base_size is not None:
        for item in annotations:
            if str(item.get("name", "")).lower() != "size":
                continue
            try:
                start = int(item.get("start", 0) or 0)
                end = int(item.get("end", 0) or 0)
            except (TypeError, ValueError):
                continue
            if (end - start) / length < 0.6:
                continue
            size = _as_size(item.get("value"))
            if size is not None and size >= 13.0 and size > base_size:
                return True

    # Аннотации есть, но ни bold ≥90%, ни size ≥13pt > base — не выделена.
    return False if has_bold_ann else None


def emphasis_from_text_markers(text: str) -> Optional[bool]:
    """Определяет emphasis по маркерам в тексте (txt/md через препроцессор).

    ``__BOLD__Текст__BOLD__`` или ``**Текст**`` → True. Маркеры присутствуют,
    но не охватывают ≥90% — False. Маркеров нет — None.
    """
    stripped = text.strip()
    if not stripped:
        return None
    # ``__BOLD__...__BOLD__`` или ``**...**`` (симметричная обёртка всей строки).
    if stripped.startswith(("__BOLD__", "**", "<b>")):
        # Симметричная обёртка всей строки = жирный заголовок.
        if stripped.startswith("__BOLD__") and stripped.endswith("__BOLD__"):
            inner = stripped[len("__BOLD__") : -len("__BOLD__")]
            return bool(inner.strip())
        if stripped.startswith("**") and stripped.endswith("**") and len(stripped) >= 4:
            inner = stripped[2:-2]
            return bool(inner.strip())
        if stripped.startswith("<b>") and stripped.endswith("</b>"):
            inner = stripped[3:-4]
            return bool(inner.strip())
    return None  # маркеров нет — источник не даёт форматирование


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "bold")


def _as_size(value: Any) -> Optional[float]:
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Проход 1: классификация строки
# --------------------------------------------------------------------------- #

#: Confidence — насколько уверены в решении.
# - hard: P0/P1/P2/P4 — фиксированные правила, не зависят от emphasis.
# - emphasis: P3/P5, принятые по bold/size.
# - chain: P3 с emphasis=None, принятый через валидацию цепочки (проход 2).
# - source: ``#``-строки Docling (md_level), не подошедшие ни под одно P.
CONFIDENCE_HARD = "hard"
CONFIDENCE_EMPHASIS = "emphasis"
CONFIDENCE_CHAIN = "chain"
CONFIDENCE_SOURCE = "source"


@dataclass
class LineClassification:
    """Результат классификации одной строки (после прохода 1).

    ``rule``: ``"S"`` (подавлено) | ``"P0_WORD"`` | ``"P0_APP"`` | ``"P0_NOTE"``
    | ``"P1"`` | ``"P2"`` | ``"P3"`` | ``"P4"`` | ``"P5"`` | ``"md"`` |
    ``None`` (не заголовок, обычный абзац).

    ``kind``: вид заголовка для реестра (``word:Раздел``, ``note``, ``arab``,
    ``roman``, ``letter``, ``caps_line``, ``md:<n>``) или ``None``.

    ``number``: числовая/буквенная часть нумерации (``"1"``, ``"1.1"``,
    ``"I"``, ``"A"``) или ``None``.

    ``title``: текст заголовка без нумерации (для отображения).

    ``confidence``: ``hard`` | ``emphasis`` | ``chain`` | ``source`` | ``None``.

    ``rank``: ранг (проставляется в проходе 3, здесь ``None``).

    ``level``: уровень в дереве (проставляется в проходе 4, здесь ``None``).

    ``suppressed``: True если строка подавлена правилом S (не заголовок).

    ``metadata``: доп. информация (``parent_number``, ``components`` и пр.).
    """

    text: str
    rule: Optional[str] = None
    kind: Optional[str] = None
    number: Optional[str] = None
    title: Optional[str] = None
    confidence: Optional[str] = None
    rank: Optional[int] = None
    level: Optional[int] = None
    suppressed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


def _passes_gates(
    title: str,
    *,
    is_numbered: bool,
    allow_colon: bool,
    allow_empty: bool = False,
    max_len: int = 150,
) -> bool:
    """Общие гейты G к тексту заголовка без нумерации.

    * длина ≤ ``max_len`` (для P3–P5 ≤100 — передаётся отдельно);
    * не оканчивается на ``. ; , : ! ?`` (исключения: сама нумерация
      «Раздел 1.», P1 с допустимым ``:``);
    * начинается с заглавной буквы или кавычки (уже проверено START в регексе,
      но для верности — здесь);
    * не дата и не «число + единица».

    :param allow_empty: для P0_APP/P0_WORD — заголовок без названия допустим
        («Приложение А (обязательное)», «Раздел 1.» — полный заголовок).
    """
    if not title:
        return allow_empty
    if len(title) > max_len:
        return False
    if _is_date_or_unit(title):
        return False
    if _ends_with_terminal_punct(title):
        if is_numbered:
            return True
        if allow_colon and title.rstrip().endswith(":"):
            return True
        return False
    return True


def _components_of(number: str) -> int:
    """Число компонентов нумерации: «1» → 1, «1.1» → 2, «1.1.1» → 3."""
    return number.count(".") + 1 if number else 1


def classify_line(
    text: str,
    *,
    source: str = "docx",
    emphasis: Optional[bool] = None,
    md_level: Optional[int] = None,
) -> LineClassification:
    """Проход 1: классификация одной строки.

    Порядок: S → P0 → P1 → P2 → P3 → P4 → P5 → md (Docling). Первое
    совпадение завершает проверку. Если ничего не сработало — абзац.

    :param text: строка текста (уже stripped).
    :param source: источник (``"docx"``, ``"txt"``, ``"md"``, ``"pdf"``).
    :param emphasis: ``True``/``False``/``None`` (см. модуль docstring).
    :param md_level: число ``#`` если строка пришла заголовком из Docling.
    """
    stripped = (text or "").strip()
    if not stripped:
        return LineClassification(text=stripped, suppressed=True)

    # --- S: подавления (проверяются первыми) ---
    # S_TOC: маркер страницы в конце.
    if S_TOC.search(stripped):
        return LineClassification(text=stripped, rule="S", suppressed=True,
                                   metadata={"s_reason": "toc"})
    # S_CAPTION.
    if S_CAPTION.match(stripped):
        return LineClassification(text=stripped, rule="S", suppressed=True,
                                   metadata={"s_reason": "caption"})
    # S_NOTE: «Примечание — текст» / «Примечание: текст» / «Примечание 1 — текст».
    # ВАЖНО: P0_NOTE (Примечание N. Название) проверяется в P0 ниже и имеет
    # приоритет — но S_NOTE срабатывает на формах БЕЗ точки после числа.
    # Чтобы не подавить P0_NOTE, проверяем S_NOTE только если НЕ подходит P0_NOTE.
    if S_NOTE.match(stripped) and not P0_NOTE.match(stripped):
        return LineClassification(text=stripped, rule="S", suppressed=True,
                                   metadata={"s_reason": "note"})
    # S_LIST.
    if S_LIST.match(stripped):
        return LineClassification(text=stripped, rule="S", suppressed=True,
                                   metadata={"s_reason": "list"})

    # --- P0: структурные слова + нумерация (ранг = base + components − 1) ---
    m = P0_WORD.match(stripped)
    if m:
        word = m.group("w")
        number = m.group("n")
        title = (m.group("t") or "").strip()
        kind = f"word:{word.capitalize()}"
        if _passes_gates(title, is_numbered=True, allow_colon=False,
                         allow_empty=True, max_len=150):
            return LineClassification(
                text=stripped, rule="P0_WORD", kind=kind, number=number,
                title=title or None, confidence=CONFIDENCE_HARD,
                metadata={"components": _components_of(number), "word": word},
            )

    # P0_APP: «Приложение А (обязательное)», «Приложение 1».
    m = P0_APP.match(stripped)
    if m:
        number = m.group("n")
        title = (m.group("t") or "").strip()
        if _passes_gates(title, is_numbered=True, allow_colon=False,
                         allow_empty=True, max_len=150):
            return LineClassification(
                text=stripped, rule="P0_APP", kind="app", number=number,
                title=title or None, confidence=CONFIDENCE_HARD,
                metadata={"components": 1, "fixed_rank": 1},
            )

    # P0_NOTE: «Примечание N. Название».
    m = P0_NOTE.match(stripped)
    if m:
        number = m.group("n")
        title = m.group("t").strip()
        if _passes_gates(title, is_numbered=True, allow_colon=False,
                         max_len=150):
            return LineClassification(
                text=stripped, rule="P0_NOTE", kind="note", number=number,
                title=title, confidence=CONFIDENCE_HARD,
                metadata={"components": 1},
            )

    # --- P1: секционные слова (фиксированный ранг 1) ---
    if P1_RU.match(stripped) or P1_EN.match(stripped):
        return LineClassification(
            text=stripped, rule="P1", kind="section", number=None,
            title=stripped, confidence=CONFIDENCE_HARD,
            metadata={"components": 1, "fixed_rank": 1},
        )

    # --- P2: нумерация + КАПС-остаток ---
    m = NUM_ARAB.match(stripped)
    if m and is_caps(m.group("t")):
        number = m.group("n")
        title = m.group("t").strip()
        if _passes_gates(title, is_numbered=True, allow_colon=False,
                         max_len=100):
            return LineClassification(
                text=stripped, rule="P2", kind="arab", number=number,
                title=title, confidence=CONFIDENCE_HARD,
                metadata={"components": _components_of(number)},
            )
    m = NUM_ROMAN.match(stripped)
    if m and is_caps(m.group("t")):
        number = m.group("n")
        title = m.group("t").strip()
        if _passes_gates(title, is_numbered=True, allow_colon=False,
                         max_len=100):
            return LineClassification(
                text=stripped, rule="P2", kind="roman", number=number,
                title=title, confidence=CONFIDENCE_HARD,
                metadata={"components": 1},
            )
    m = NUM_LETTER.match(stripped)
    if m and is_caps(m.group("t")) and not INITIALS_RE.match(stripped):
        number = m.group("n")
        title = m.group("t").strip()
        if _passes_gates(title, is_numbered=True, allow_colon=False,
                         max_len=100):
            return LineClassification(
                text=stripped, rule="P2", kind="letter", number=number,
                title=title, confidence=CONFIDENCE_HARD,
                metadata={"components": 1},
            )

    # --- P3: арабская нумерация, t не КАПС, ≤100, emphasis-зависимое ---
    m = NUM_ARAB.match(stripped)
    if m:
        number = m.group("n")
        title = m.group("t").strip()
        if _passes_gates(title, is_numbered=True, allow_colon=False,
                         max_len=100):
            components = _components_of(number)
            # emphasis True → принять. False → отклонить. None → только формы
            # ``N Название`` и ``N.M Название`` (не ``N)``); одиночное ``N.``
            # не принимать; валидация цепочки — в проходе 2.
            sep = m.group("sep")
            if emphasis is True:
                return LineClassification(
                    text=stripped, rule="P3", kind="arab", number=number,
                    title=title, confidence=CONFIDENCE_EMPHASIS,
                    metadata={"components": components, "sep": sep},
                )
            if emphasis is None:
                # ``N)`` отклоняется (ветка перечня); ``N.`` без названия —
                # отклоняется; ``N Название`` и ``N.M Название`` — кандидаты
                # для прохода 2 (chain).
                if sep == ")" or (sep == "." and not title):
                    return LineClassification(text=stripped)
                # Кандидат chain — confidence проставится в проходе 2.
                return LineClassification(
                    text=stripped, rule="P3", kind="arab", number=number,
                    title=title, confidence=None,  # ждёт валидации цепочки
                    metadata={"components": components, "sep": sep,
                              "chain_candidate": True},
                )
            # emphasis False → отклонить.
            return LineClassification(text=stripped)

    # --- P4: нет нумерации, is_caps(строка), ≤100, без терминальной пунктуации ---
    # Отсекаем инициалы «А. С. ПУШКИН» — это не заголовок, а ФИО.
    if (is_caps(stripped) and len(stripped) <= 100
            and not _ends_with_terminal_punct(stripped)
            and not INITIALS_RE.match(stripped)):
        return LineClassification(
            text=stripped, rule="P4", kind="caps_line", number=None,
            title=stripped, confidence=CONFIDENCE_HARD,
            metadata={"components": 1},
        )

    # --- P5: римская/буквенная нумерация, t не КАПС, ≤60, только emphasis=True ---
    if emphasis is True:
        m = NUM_ROMAN.match(stripped)
        if m and not is_caps(m.group("t")):
            number = m.group("n")
            title = m.group("t").strip()
            if _passes_gates(title, is_numbered=True, allow_colon=False,
                             max_len=60):
                return LineClassification(
                    text=stripped, rule="P5", kind="roman", number=number,
                    title=title, confidence=CONFIDENCE_EMPHASIS,
                    metadata={"components": 1},
                )
        m = NUM_LETTER.match(stripped)
        if m and not is_caps(m.group("t")) and not INITIALS_RE.match(stripped):
            number = m.group("n")
            title = m.group("t").strip()
            if _passes_gates(title, is_numbered=True, allow_colon=False,
                             max_len=60):
                return LineClassification(
                    text=stripped, rule="P5", kind="letter", number=number,
                    title=title, confidence=CONFIDENCE_EMPHASIS,
                    metadata={"components": 1},
                )

    # --- md: ``#``-строки Docling (после S и G) ---
    if md_level is not None and md_level >= 1:
        return LineClassification(
            text=stripped, rule="md", kind=f"md:{md_level}", number=None,
            title=stripped, confidence=CONFIDENCE_SOURCE,
            metadata={"components": 1, "md_level": md_level},
        )

    # Ничего не сработало — обычный абзац.
    return LineClassification(text=stripped)


# --------------------------------------------------------------------------- #
# Проход 2: валидация нумерации (цепочки)
# --------------------------------------------------------------------------- #

def _next_in_chain(prev: str, current: str) -> bool:
    """Допустимый ли преемник ``current`` для ``prev`` в цепочке нумерации.

    Преемники ``a.b.c``:
      * ``a.b.(c+1)`` — сосед (та же длина, последний +1);
      * ``a.b.c.1`` — углубление (длина +1, последний = 1);
      * ``a.(b+1)`` — возврат на уровень вверх (длина −1, последний +1);
      * ``(a+1)`` — новый корневой раздел (длина 1, +1).
    """
    if not prev or not current:
        return False
    p = [int(x) for x in prev.split(".") if x.isdigit()]
    c = [int(x) for x in current.split(".") if x.isdigit()]
    if not p or not c:
        return False
    # (a+1): current = [p[0]+1], длина 1.
    if len(c) == 1 and len(p) == 1 and c[0] == p[0] + 1:
        return True
    if len(c) == 1 and len(p) > 1 and c[0] == p[0] + 1:
        # «2» после «1.1.1» — новый корневой раздел.
        return True
    # a.b.(c+1): current длина = len(p), последний +1, остальные совпадают.
    if len(c) == len(p) and c[:-1] == p[:-1] and c[-1] == p[-1] + 1:
        return True
    # a.b.c.1: current длина = len(p)+1, префикс = p, последний = 1.
    if len(c) == len(p) + 1 and c[:-1] == p and c[-1] == 1:
        return True
    # a.(b+1): current длина = len(p)-1, префикс = p[:-2], последний = p[-2]+1.
    # Пример: prev=[1,1,1] → current=[1,2] (возврат на уровень вверх).
    if len(p) >= 2 and len(c) == len(p) - 1 and c[:-1] == p[:-2] and c[-1] == p[-2] + 1:
        return True
    return False


def validate_chains(
    classifications: List[LineClassification],
) -> List[LineClassification]:
    """Проход 2: валидация нумерации для P3-chain и P2_LETTER кандидатов.

    Для подряд идущих кандидатов (``chain_candidate=True`` или
    ``kind=="letter"`` с confidence ``hard``) строится цепочка, где каждый
    элемент — допустимый преемник предыдущего. Кандидат принимается, если
    входит в цепочку длиной ≥2 (многоуровневые) или ≥3 (одноуровневые).

    Отклонённые остаются абзацами (``rule=None``, счётчик ``demoted_by_chain``
    в возвращаемом списке через metadata корневого элемента — нет, проще
    считать в ``build_tree``).
    """
    # Собираем индексы кандидатов.
    candidates: List[int] = []
    for i, c in enumerate(classifications):
        if c.suppressed or c.rule is None:
            continue
        # P3-chain (confidence None, chain_candidate=True).
        if c.metadata.get("chain_candidate"):
            candidates.append(i)
        # P2_LETTER с confidence hard — тоже валидируется цепочкой
        # (A, B, C… — нужна серия ≥3).
        elif c.rule == "P2" and c.kind == "letter":
            candidates.append(i)

    if not candidates:
        return classifications

    # Группируем подряд идущих кандидатов в раны.
    runs: List[List[int]] = []
    current_run: List[int] = []
    for idx in candidates:
        if not current_run or idx == current_run[-1] + 1:
            current_run.append(idx)
        else:
            if current_run:
                runs.append(current_run)
            current_run = [idx]
    if current_run:
        runs.append(current_run)

    for run in runs:
        # Для каждого кандидата в ране — проверяем, входит ли он в цепочку.
        # Цепочка = последовательность, где каждый следующий — преемник.
        # Начало цепочки: «1» (для arab) или первая буква (для letter).
        accepted_in_run: List[int] = []
        prev_number: Optional[str] = None
        for idx in run:
            c = classifications[idx]
            num = c.number or ""
            if prev_number is None:
                # Начало цепочки: «1» (arab) или любая буква (letter).
                if c.kind == "arab" and num == "1":
                    accepted_in_run.append(idx)
                    prev_number = num
                elif c.kind == "letter":
                    # Буквенная цепочка — старт с любой, но нужна серия ≥3.
                    accepted_in_run.append(idx)
                    prev_number = num
                else:
                    continue
            else:
                if _next_in_chain(prev_number, num):
                    accepted_in_run.append(idx)
                    prev_number = num
                else:
                    # Разрыв цепочки — но возможен старт новой.
                    if c.kind == "arab" and num == "1":
                        accepted_in_run.append(idx)
                        prev_number = num
                    elif c.kind == "letter":
                        accepted_in_run.append(idx)
                        prev_number = num
                    else:
                        prev_number = None

        # Минимальная длина цепочки: ≥2 (многоуровневые) или ≥3 (одноуровневые).
        # Одноуровневая = все элементы одной длины (только «1», «2», «3»).
        if len(accepted_in_run) < 2:
            # Демоутим ВЕСЬ раны (не только accepted_in_run) — цепочка слишком коротка.
            for idx in run:
                classifications[idx] = _demote(classifications[idx])
            continue
        # Проверим, есть ли многоуровневые (разной длины).
        lengths = {classifications[idx].metadata.get("components", 1)
                   for idx in accepted_in_run}
        is_multi = len(lengths) > 1
        if not is_multi and len(accepted_in_run) < 3:
            # Одноуровневая цепочка <3 — демоутим весь раны.
            for idx in run:
                classifications[idx] = _demote(classifications[idx])
            continue
        # Принимаем кандидатов — confidence = chain. Те, кто не попал в
        # accepted_in_run (разрыв цепочки внутри раны), тоже демоутим.
        accepted_set = set(accepted_in_run)
        for idx in run:
            if idx in accepted_set:
                classifications[idx].confidence = CONFIDENCE_CHAIN
            else:
                classifications[idx] = _demote(classifications[idx])

    return classifications


def _demote(c: LineClassification) -> LineClassification:
    """Понижает кандидата chain до абзаца (rule=None)."""
    return LineClassification(text=c.text, rule=None, suppressed=False)


# --------------------------------------------------------------------------- #
# Проход 3: ранги (реестр видов)
# --------------------------------------------------------------------------- #

#: Виды с фиксированным рангом 1 (P0_APP, P1).
_FIXED_RANK_KINDS = {"app", "section"}


def assign_ranks(
    classifications: List[LineClassification],
) -> List[LineClassification]:
    """Проход 3: назначение рангов по реестру видов.

    * ``app`` и ``section`` (P0_APP, P1) — фиксированный ранг 1.
    * Остальные виды: ``base(kind)`` — ранг вершины стека среди заголовков
      не с фиксированным рангом + 1, при пустом стеке — 1. ``base`` считается
      один раз при первом появлении вида и больше не меняется.
    * Нумерованные заголовки: ранг = ``base(kind) + components − 1``
      (но не глубже, см. проход 4 — там стек обрежет).

    «Введение» и другие P1 при вычислении base родителем не считаются.
    """
    base_by_kind: Dict[str, int] = {}
    # Стек видов (не fixed-rank), для вычисления base.
    kind_stack: List[str] = []

    for c in classifications:
        if c.suppressed or c.rule is None:
            continue
        kind = c.kind or ""
        components = c.metadata.get("components", 1)

        if kind in _FIXED_RANK_KINDS:
            c.rank = 1
            # P1 (Введение/Содержание/Приложение) начинает новый раздел
            # документа — стек видов сбрасывается, чтобы следующие заголовки
            # считали base с пустого стека (спецификация: «Введение и другие
            # P1 при вычислении base родителем не считаются»).
            kind_stack.clear()
            continue

        # dedoc-заголовки (стиль Heading) — ранг = уровень из dedoc.
        if kind == "dedoc":
            c.rank = c.metadata.get("dedoc_level", 1)
            continue

        # base(kind) ещё не назначен — вычисляем.
        if kind not in base_by_kind:
            # Ранг вершины стека среди не-fixed + 1, при пустом — 1.
            if kind_stack:
                top_kind = kind_stack[-1]
                top_rank = base_by_kind.get(top_kind, 1)
                base = top_rank + 1
            else:
                base = 1
            base_by_kind[kind] = base
        else:
            base = base_by_kind[kind]

        # Нумерованные: ранг = base + components − 1.
        if c.number:
            c.rank = base + components - 1
        else:
            # caps_line, md:<n> — ранг = base (без нумерации, нет компонентов).
            # Для md:<n> — специальная обработка: ранг = md_level (если вид
            # ещё не в реестре, base = md_level; иначе base).
            if c.rule == "md":
                md_level = c.metadata.get("md_level", 1)
                if kind not in base_by_kind or base_by_kind.get(kind) == 1:
                    # Первый md-заголовок задаёт base = md_level.
                    base_by_kind[kind] = md_level
                    base = md_level
                c.rank = base
            else:
                c.rank = base

        # Кладём вид в стек (только не-fixed-rank).
        if kind not in _FIXED_RANK_KINDS:
            # Снимаем со стека виды с рангом ≥ текущего base.
            while kind_stack and base_by_kind.get(kind_stack[-1], 1) >= base:
                kind_stack.pop()
            kind_stack.append(kind)

    return classifications


# --------------------------------------------------------------------------- #
# Проход 4: дерево (стек)
# --------------------------------------------------------------------------- #

#: Максимальная глубина (из README — 3, скользящее окно).
MAX_DEPTH = 3


def build_tree(
    classifications: List[LineClassification],
) -> Tuple[List[LineClassification], Dict[str, Any]]:
    """Проход 4: построение дерева со стеком + назначение level.

    Один проход: пришёл заголовок ранга r → снять со стека всё с рангом ≥ r →
    вершина = родитель → положить в стек. Уровень = глубина в дереве
    (ограничено ``MAX_DEPTH``).

    :return: ``(classifications, stats)`` где stats — счётчики по правилам,
        ``demoted_by_chain``, ``numbering_gaps``.
    """
    stack: List[Tuple[LineClassification, int]] = []  # (classification, depth)
    stats: Dict[str, Any] = {
        "by_rule": {},
        "by_kind": {},
        "demoted_by_chain": 0,
        "numbering_gaps": 0,
        "total_headers": 0,
    }

    for c in classifications:
        if c.suppressed or c.rule is None:
            continue
        # Считаем stats.
        stats["by_rule"][c.rule] = stats["by_rule"].get(c.rule, 0) + 1
        stats["by_kind"][c.kind or ""] = stats["by_kind"].get(c.kind or "", 0) + 1
        if c.confidence == CONFIDENCE_CHAIN:
            pass  # chain принят
        elif c.metadata.get("chain_candidate") and c.confidence != CONFIDENCE_CHAIN:
            stats["demoted_by_chain"] += 1
        stats["total_headers"] += 1

        rank = c.rank or 1
        # Снимаем со стека всё с рангом ≥ текущего.
        while stack and stack[-1][0].rank >= rank:
            stack.pop()
        # Глубина = длина стека + 1, ограничена MAX_DEPTH.
        depth = min(len(stack) + 1, MAX_DEPTH)
        c.level = depth
        # Родитель — вершина стека (если есть).
        c.metadata["parent_number"] = stack[-1][0].number if stack else None
        # Проверка numbering_gap для нумерованных.
        if stack and c.number and stack[-1][0].number:
            parent_num = stack[-1][0].number
            if not _is_prefix(parent_num, c.number):
                stats["numbering_gaps"] += 1
        # Кладём в стек.
        stack.append((c, depth))

    return classifications, stats


def _is_prefix(parent: str, child: str) -> bool:
    """``parent`` — префикс ``child`` по нумерации («1» → «1.1», «1.1» → «1.1.1»)."""
    if not parent or not child:
        return False
    p = parent.split(".")
    c = child.split(".")
    if len(c) <= len(p):
        return False
    return c[: len(p)] == p


__all__ = [
    "ARAB",
    "CONFIDENCE_CHAIN",
    "CONFIDENCE_EMPHASIS",
    "CONFIDENCE_HARD",
    "CONFIDENCE_SOURCE",
    "LineClassification",
    "MAX_DEPTH",
    "NUM_ARAB",
    "NUM_LETTER",
    "NUM_ROMAN",
    "P0_APP",
    "P0_NOTE",
    "P0_WORD",
    "P1_EN",
    "P1_RU",
    "ROMAN",
    "S_CAPTION",
    "S_LIST",
    "S_NOTE",
    "S_TOC",
    "assign_ranks",
    "build_tree",
    "classify_line",
    "emphasis_from_annotations",
    "emphasis_from_text_markers",
    "is_caps",
    "validate_chains",
]
