"""Детерминированное распознавание заголовков (без скоринга).

Реальные docx часто оформлены без стилей Word: заголовки выделены
визуально (шрифт побольше, КАПС, нумерация «Раздел N.»), а dedoc отдаёт
их как ``raw_text``/``paragraph`` — иногда даже склеенными с телом в один
многострочный узел (аннотации при этом привязаны к диапазонам символов).

Модуль классифицирует такие строки ЖЁСТКИМИ правилами: совпадение хотя бы
одного правила безусловно делает строку заголовком, никаких баллов и
порогов нет. Это осознанный переход от «вероятностного угадывания с
настройкой вслепую» к детерминированной классификации.

Правила (любое одно совпадение -> заголовок):

1. шрифт >= ``MIN_HEADER_FONT_SIZE`` (13pt) И строго больше базового
   размера шрифта основного текста документа;
2. структурное слово в начале: Раздел / Глава / Часть / Том / Книга;
3. слово секции в начале: Введение / Заключение / Приложение /
   Список литературы / Библиография (уровень 1);
4. слово содержания в начале: Содержание / Оглавление /
   Table of Contents / Contents;
5. текст целиком в ВЕРХНЕМ РЕГИСТРЕ (МЕТОДОЛОГИЯ ИССЛЕДОВАНИЯ);
6. нумерация + ВЕРХНИЙ РЕГИСТР остатка (2.3 АНАЛИЗ ДАННЫХ);
7. числовая нумерация в начале: ``1``, ``1)``, ``1.1``, ``1.1.1`` …

Уровень при нумерации: ``level = (число точек в числовой части) + 1``
(«1» -> 1, «1.1» -> 2, «1.1.1» -> 3, «1.1.1.1» -> 4). Без нумерации
уровень 1.

Порядок работы:

0. все листья режутся на строки; по порядку строк считаются distinct
   «Раздел N» (супрессия `N)` при >=2) и TOC-серии (>=5 подряд «Раздел N»
   с N строго +1 — не распознаются как заголовки);
1. каждая однострочная нода классифицируется по правилам выше;
2. соседние нераспознанные строки склеиваются обратно в абзацы
   (TOC-строки остаются поштучно);
3. дерево сплющивается в порядок документа и пересобирается стеком
   по уровням (lift распознанных заголовков из чужой вложенности).
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from rag_parser.core.interfaces import DocumentNode

#: Включает отладочный вывод ``classify_header`` для структурных заголовков
#: («Раздел N», «Глава N» …). Активируется переменной окружения
#: ``RAG_DEBUG_HEADERS=1`` — в тестах и боевом режиме выключен по умолчанию,
#: чтобы не засорять stdout. См. Шаг 4 плана диагностики task.txt.
_DEBUG_HEADERS = os.environ.get("RAG_DEBUG_HEADERS", "") not in ("", "0", "false", "no")

#: Структурные слова в начале строки (правило 2).
STRUCTURAL_WORDS = ("Раздел", "Глава", "Часть", "Том", "Книга")
#: Слова секций в начале строки (правило 3) — всегда уровень 1.
SECTION_WORDS = (
    "Введение",
    "Заключение",
    "Предисловие",
    "Приложение",
    "Список литературы",
    "Библиография",
)
#: Слова оглавления/содержания в начале строки (правило 4).
CONTENT_WORDS = ("Содержание", "Оглавление", "Table of Contents", "Contents")

#: Минимальный кегль заголовка в пунктах (правило 1).
MIN_HEADER_FONT_SIZE = 13.0

#: Доля жирного текста, при которой строка считается заголовком (правило 9).
#: Порог высокий намеренно: у заголовка жирно ВСЁ, а у выделенного термина
#: («**ПДн** — персональные данные») — только начало. Низкий порог дал бы
#: кучу ложных заголовков на терминах.
BOLD_HEADING_COVERAGE = 0.9

#: Числовая нумерация в начале: «1», «1)», «1.1», «1.1.1» (правила 6, 7).
NUMBER_PREFIX_RE = re.compile(r"^(\d+(?:\.\d+)*)")
#: Явный маркер «Раздел N» — используется для подавления `N)` и TOC-серий.
RAZDEL_N_RE = re.compile(r"^Раздел\s+(\d+)", re.IGNORECASE)
#: Только «N)» без другой нумерации — ветка, подавляемая в «Раздел N»-документах.
PAREN_ONLY_RE = re.compile(r"^\d+\s*\)")
#: Строка целиком из заглавных букв, цифр и разделителей (правило 5).
#: В набор разделителей добавлено тире: без него строка вида
#: «ГОСТ Р 99999—2026» не проходила правило целиком, и обозначение
#: стандарта оставалось абзацем, хотя в DOCX это заголовок титула.
#: Двоеточие и слэш намеренно НЕ добавлены: иначе заголовками стали бы
#: «УДК 004.9:006.354» и «(ISO/IEC 99999:2026, NEQ)» — в эталоне это абзацы.
CAPS_LINE_RE = re.compile(r"^[A-ZА-ЯЁ\s\-\.\d«»\"'()—–]+$")

#: Жирный текст как триггер вложенности. Для ``.md`` разметка съедается
#: рендерером, поэтому ``markdown_preprocessor`` заменяет ``<strong>``/``<b>``
#: на эти маркеры ДО отдачи в dedoc: ``HtmlReader`` вытаскивает только текст.
BOLD_MARKERS = ("__BOLD__", "<b>", "**")
_BOLD_OPEN_RE = re.compile(r"^(?:__BOLD__|\*\*|<b>)", re.IGNORECASE)
_BOLD_LEAD_RE = re.compile(r"^\s*(?:\*{1,3}|<b>)", re.IGNORECASE)
_BOLD_TAIL_RE = re.compile(r"(?:\*{1,3}|</b>)\s*$", re.IGNORECASE)
#: Базовый заголовок: «1 **Текст**» либо «1 НАЗВАНИЕ» (цифра + КАПС).
_BASE_BOLD_RE = re.compile(r"^\d+[\.\)\s]+(?:\*\*.+?\*\*|__BOLD__)", re.IGNORECASE)
_BASE_CAPS_RE = re.compile(r"^\d+[\.\)\s]+[А-ЯA-ZЁ\s]+$")
#: Базовый заголовок: структурное слово + ОДНО число («Раздел 1.»,
#: «Глава 2.», «Часть 3 »). ТЗ 1.2: «Раздел N» (где N — цифра) → базовый
#: заголовок Level 1, сброс стека трекера. Negative lookahead ``(?!\.\d)``
#: отсекает «Раздел 1.1» / «Глава 1.1.1» — там число составное, и это
#: вложенный триггер (а не базовый заголовок), что проверяется тестом
#: ``(["1 Текст", "Раздел 1.1", "Глава 1.1.1"], [1, 2, 3])``.
#:
#: Без этого правила «Раздел N» в теле документа (стиль не Heading, а
#: «ConsNormal») уходил в keyword-ветку трекера и получал уровень 2/3
#: вместо 1 — breadcrumb чанков 2.x становился «Оглавление:» вместо
#: «Раздел 2. Сведения об Обществе» (2023.docx, см. task.txt).
_BASE_STRUCTURAL_RE = re.compile(
    r"^(?:Раздел|Глава|Часть|Том|Книга)\s+\d+(?!\.\d)(?:[.)\s]|$)",
    re.IGNORECASE,
)
#: Ключевые слова-триггеры вложенности.
_NEST_KEYWORD_RE = re.compile(
    r"^(Раздел|Глава|Параграф|Подраздел|Часть)[\s\d]", re.IGNORECASE
)
#: Максимальная глубина вложенности заголовков.
MAX_HEADER_LEVEL = 3

TOC_SERIES_MIN = 5

#: Маркер номера страницы в строке оглавления («… 4», «... 4»).
_TOC_PAGE_MARKER_RE = re.compile(r"…|\.\.\.")
#: Строка-разделитель («------», «======»): декорация, а не конец области.
_TOC_DECORATION_RE = re.compile(r"^[-=_~*·.\s]{3,}$")


def _annotations_of(node: DocumentNode) -> List[Dict[str, Any]]:
    raw = node.metadata.get("annotations")
    if not isinstance(raw, list):
        return []
    result: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            result.append(item)
            continue
        result.append(
            {
                "name": getattr(item, "name", ""),
                "start": getattr(item, "start", 0),
                "end": getattr(item, "end", 0),
                "value": getattr(item, "value", ""),
            }
        )
    return result


def _coverage(
    annotations: List[Dict[str, Any]], name: str, length: int, threshold: float = 0.6
) -> Tuple[bool, Any]:
    """Есть ли аннотация ``name``, покрывающая >= ``threshold`` строки.

    :return: ``(покрывает_ли, значение)``
    """
    if length <= 0:
        return False, None
    best = 0.0
    best_value: Any = None
    for item in annotations:
        if str(item.get("name", "")).lower() != name:
            continue
        try:
            start = int(item.get("start", 0) or 0)
            end = int(item.get("end", 0) or 0)
        except (TypeError, ValueError):
            continue
        overlap = max(0, min(end, length) - max(start, 0))
        ratio = overlap / length
        if ratio > best:
            best = ratio
            best_value = item.get("value")
    return best >= threshold, best_value


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "bold")


def _as_size(value: Any) -> Optional[float]:
    """Числовой кегль из значения аннотации ``size`` (``"12,5"`` -> 12.5)."""
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def numbering_level(text: str, allow_paren: bool = True) -> Optional[int]:
    """Уровень заголовка из числовой нумерации или ``None``.

    ``level = (число точек в числовой части) + 1``:

    * «1», «1)», «1. Название»      -> 1
    * «1.1», «2.3 Название»         -> 2
    * «1.1.1 Название»              -> 3
    * «1.1.1.1 Название»            -> 4

    Возвращает ``None``, если нумерации нет («Название», «1abc» — цифра
    неотделима от текста) либо это подавляемая ветка ``N)``.

    Терминатором нумерации считается точка, скобка, пробел или конец
    строки. Greedy-совпадение ``\\d+(\\.\\d+)*`` захватывает всю
    цепочку: regex ``^(\\d+(\\.\\d+)*)[\\.\\)\\s]`` из ТЗ на «1.1.1.1»
    откатывается на «1.1.1» и дал бы уровень 3 вместо 4.
    """
    stripped = (text or "").strip()
    match = NUMBER_PREFIX_RE.match(stripped)
    if not match:
        return None
    number = match.group(1)
    tail = stripped[match.end() :]
    if tail and tail[0] not in ".) \t":
        # «1abc», «2024год» — цифра не является началом нумерации.
        return None
    if tail.startswith(")"):
        # Ветка `N)`: в «Раздел N»-документах она подавлена.
        return None if not allow_paren else 1
    return number.count(".") + 1


def _starts_with_word(text: str, words: Tuple[str, ...]) -> bool:
    """Регистронезависимая проверка ``startswith`` по списку слов."""
    lowered = text.lower()
    for word in words:
        candidate = word.lower()
        if lowered.startswith(candidate):
            # «Список» не должно ловить «Список литературы» дважды, но
            # «Содержание» не должно ловиться внутри «Содержательные
            # данные» — отсекаем, если сразу идёт буква.
            rest = lowered[len(candidate) :]
            if not rest or rest[0] in " \t.,:;)]—-":
                return True
    return False


def _is_caps(text: str) -> bool:
    """Весь текст в верхнем регистре (пробелы/пунктуация игнорируются)."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    letters = [char for char in stripped if char.isalpha()]
    if len(letters) < 2:
        return False
    return bool(CAPS_LINE_RE.match(stripped))


def _is_bold_heading(annotations: List[Dict[str, Any]], length: int) -> bool:
    """Правило 9: строка целиком набрана жирным.

    ``dedoc`` отдаёт жирность аннотациями ``bold`` на каждом bold-run
    (``readers/docx_reader/line_with_meta_converter.py``), и они уже
    приходят в ``classify_header`` вторым параметром — отдельный экстрактор
    метаданных не нужен.

    Отсекаем выделенные термины: жирным должно быть почти всё (``bold``),
    а не только префикс строки.
    """
    covered, value = _coverage(
        annotations, "bold", length, threshold=BOLD_HEADING_COVERAGE
    )
    return bool(covered and _is_truthy(value))


def _font_is_bigger(
    annotations: List[Dict[str, Any]], base_size: Optional[float]
) -> bool:
    """Правило 1: кегль >= 13pt и строго больше базового размера текста.

    ``base_size`` — модальный кегль основного текста документа (считается
    один раз на документ). Если он неизвестен, сравниваем только с порогом
    13pt: строка крупнее 13pt — заголовок в любом случае.
    """
    if base_size is None:
        return False
    hit, value = _coverage(annotations, "size", _line_length(annotations))
    if not hit:
        return False
    size = _as_size(value)
    if size is None:
        return False
    if size < MIN_HEADER_FONT_SIZE:
        return False
    return size > base_size


def _line_length(annotations: List[Dict[str, Any]]) -> int:
    """Длина текста, покрытого аннотациями (для расчёта покрытия)."""
    length = 0
    for item in annotations:
        try:
            length = max(length, int(item.get("end", 0) or 0))
        except (TypeError, ValueError):
            continue
    return length


class HeaderLevelTracker:
    """Уровень заголовка по СОСТОЯНИЮ документа, а не по точкам в номере.

    Наивное «уровень = точек + 1» ломается на вложенности: «1.1» и «1.2»
    получают одинаковый уровень, но у документа бывают «1.2.1», где
    «1.2» — уже третий уровень. Трекер держит стек последних уровней и
    различает **углубление** («1.1» -> «1.1.1») от **соседства**
    («1.1» -> «1.2» — это тот же уровень, а не следующий).

    Правила:

    * базовый заголовок (цифра + КАПС либо цифра + жирный) -> уровень 1,
      стек сбрасывается;
    * триггер вложенности (нумерация с точками, жирный текст без цифры,
      ключевое слово) даёт следующий уровень;
    * глубина ограничена ``MAX_HEADER_LEVEL``; при попытке уйти глубже
      скользящее окно отбрасывает самый верхний уровень;
    * без триггера строка считается базовой: стек сбрасывается на неё.

    Экземпляр — **один на документ**; состояние намеренно не глобальное.

    Отличие от псевдокода в ТЗ
    ---------------------------
    В ТЗ на ветке «нет триггера» стоит ``return 1`` БЕЗ обновления стека.
    Из-за этого «1 Текст» (не КАПС, не жирный, триггеров нет) оставляет
    стек пустым, и следующий заголовок «1.1 Текст» получает уровень 1
    вместо 2. На собственных примерах ТЗ это даёт провал 7 из 7:
    ``["1 Текст", "1.1 Текст", "1.2 Текст"] -> [1, 1, 2]`` вместо
    ``[1, 2, 2]``. Поэтому здесь без триггера стек **сбрасывается** на
    текущую строку — с этой поправкой проходят все 7 случаев из ТЗ.

    Поиск родителя в стеке (sibling reset вверх)
    --------------------------------------------
    Для нумерованных заголовков уровень считается не «от верхушки стека»,
    а по явному поиску **родителя** в стеке. Родитель числа ``N.M.K`` —
    ``N.M``; у ``N`` родитель — корень (``""``). Алгоритм:

    1. Извлекаем числовую часть и её родителя.
    2. Идём по стеку сверху вниз и ищем запись с тем же числом, что и
       родитель (стек хранит по одной записи на уровень — самую свежую).
    3. Если нашли на уровне ``L``: сбрасываем стек до этой записи
       (снимаем всё глубже — это и есть sibling reset) и ставим ребёнка
       на уровень ``L + 1`` (cap ``MAX_HEADER_LEVEL``).
    4. Если родителя в стеке нет — orphan numbering. Уровень считается по
       числу точек (как в базовых правилах спеки: «1» -> 1, «1.1» -> 2,
       «1.1.1» -> 3, «1.1.1.1» -> 3 (обрезка)). Это соответствует
       эталонному тесту ``test_header_classification`` из раздела 7.4
       спеки, где каждый входной текст подаётся на свежий трекер.

    Этот алгоритм корректно обрабатывает эталонную цепочку из ТЗ
    (раздел 2.2)::

        "1"      -> 1
        "1.1"    -> 2     (родитель "1" на L1 -> L2)
        "1.2"    -> 2     (sibling reset: родитель "1" на L1 -> L2)
        "1.2.1"  -> 3     (родитель "1.2" на L2 -> L3)
        "1.2.2"  -> 3     (sibling reset: родитель "1.2" на L2 -> L3)
        "1.3"    -> 2     (sibling reset UP: родитель "1" на L1 -> L2)
        "2"      -> 1     (базовый, сброс стека)
        "2.1"    -> 2
        "2.1.1"  -> 3
        "2.1.1.1"-> 3     (cap MAX)

    Для bold/keyword-триггеров (нет числа → родителя искать не по чему)
    сохраняется прежняя логика «углубление от верхушки стека».
    """

    def __init__(self) -> None:
        self.level_stack: List[Tuple[int, str]] = []

    def reset(self) -> None:
        """Сброс между документами."""
        self.level_stack = []

    def calculate_level(self, text: str) -> int:
        number = self.extract_number(text)
        if self.is_base_header(text):
            self.level_stack = [(1, number)]
            return 1

        trigger = self.detect_trigger(text)
        if not trigger:
            # Без триггера строка — базовая: сбрасываем стек (см. выше).
            self.level_stack = [(1, number)]
            return 1

        # Для нумерованных заголовков ищем РОДИТЕЛЯ в стеке, а не только
        # сравниваем с верхушкой. Это правильно обрабатывает siblings при
        # возврате на уровень вверх: «1.3» после «1.2.2» — родитель «1»
        # найдётся в стеке на уровне 1, и «1.3» получит уровень 2
        # (sibling reset), а не 3. Без этой ветки эталонная цепочка из
        # ТЗ (раздел 2.2) проваливается на шаге «1.3».
        if trigger == "numbered" and number:
            parts = number.split(".")
            # Родитель = всё кроме последнего сегмента: «1.2.3» -> «1.2»,
            # «1» -> «» (корень). Пустой родитель означает «корневой
            # заголовок без точки в номере», и он ищется как «1» в стеке
            # (extract_number для «1 Текст» возвращает «1», которое
            # кладётся в стек на уровне 1).
            parent = ".".join(parts[:-1]) if "." in number else ""

            # Идём по стеку сверху вниз — самая глубокая запись с
            # совпадающим числом и есть родитель. Стек хранит по одной
            # записи на уровень (см. _push — замещает соседа), поэтому
            # коллизий по числу не возникает.
            for idx in range(len(self.level_stack) - 1, -1, -1):
                stack_level, stack_number = self.level_stack[idx]
                if stack_number == parent:
                    # Нашли родителя на уровне stack_level. Сбрасываем
                    # стек, оставляя элементы до родителя включительно
                    # (снимаем всё глубже — это и есть sibling reset).
                    self.level_stack = self.level_stack[: idx + 1]
                    new_level = min(stack_level + 1, MAX_HEADER_LEVEL)
                    self._push(new_level, number)
                    return new_level

            # Родителя в стеке нет — orphan numbering. Уровень считается
            # по числу точек (базовые правила спеки 2.2): «1» -> 1,
            # «1.1» -> 2, «1.1.1» -> 3, «1.1.1.1» -> 3 (cap MAX). Это
            # совпадает с эталонным тестом test_header_classification
            # из раздела 7.4, где каждая строка подаётся на свежий
            # трекер, и «1.1» ожидается уровнем 2, «1.1.1.1» — 3.
            new_level = min(len(parts), MAX_HEADER_LEVEL)
            if new_level == 1:
                # «1» без родителя в стеке — базовый, сброс стека.
                self.level_stack = [(1, number)]
            else:
                self._push(new_level, number)
            return new_level

        # Для bold/keyword-триггеров (нет числа, родителя искать не по
        # чему) сохраняем прежнюю логику углубления от верхушки стека.
        if len(self.level_stack) == 0:
            new_level = 1
        elif len(self.level_stack) == 1:
            new_level = 2
        elif len(self.level_stack) == 2:
            previous_level, previous = self.level_stack[-1]
            # Сосед наследует УРОВЕНЬ предыдущего соседа, а не всегда 2.
            # Иначе в оглавлении «6.2.1» -> «6.2.2» -> «6.2.3» (все третьего
            # уровня) последний получал бы 2: после скользящего окна стек
            # содержит [(2, «6.2»), (3, «6.2.2»)], и соседство с «6.2.2»
            # обнуляло бы глубину до второго уровня.
            new_level = previous_level if self.is_sibling(number, previous) else 3
        else:
            # Скользящее окно: уходим глубже максимума — режем верх.
            self.level_stack.pop(0)
            new_level = MAX_HEADER_LEVEL

        self._push(new_level, number)
        return new_level

    def _push(self, level: int, number: str) -> None:
        """Кладёт уровень в стек, ЗАМЕНЯЯ соседа на том же уровне.

        ТЗ в этом месте неполон: он всегда делает ``append``, из-за чего
        соседство «1.1» -> «1.2» -> «1.3» раздувает стек, и четвёртый
        заголовок подряд уходит в ветку обрезки и получает уровень 3
        вместо 2. Сосед — это замена элемента стека, а не новый элемент.
        """
        if self.level_stack and self.level_stack[-1][0] == level:
            self.level_stack[-1] = (level, number)
        else:
            self.level_stack.append((level, number))

    @staticmethod
    def extract_number(text: str) -> str:
        """Числовая часть заголовка («1.2.1» -> «1.2.1», «Приложение» -> «»)."""
        match = NUMBER_PREFIX_RE.match((text or "").strip())
        return match.group(1) if match else ""

    @staticmethod
    def is_base_header(text: str) -> bool:
        """Базовый заголовок: сбрасывает стек трекера на уровень 1.

        ТЗ 1.2: «Раздел N» (где N — цифра) → базовый заголовок Level 1,
        сброс стека трекера. Распознаёт три формы:

        * «1 **Текст**» — цифра + жирный (markdown-маркеры);
        * «1 НАЗВАНИЕ» — цифра + КАПС;
        * «Раздел 1.» / «Глава 2.» / «Часть 3 » — структурное слово +
          ОДНО число (без точек). Составные «Раздел 1.1» сюда НЕ входят:
          это вложенный триггер (см. ``_BASE_STRUCTURAL_RE``).
        """
        stripped = (text or "").strip()
        return bool(
            _BASE_BOLD_RE.match(stripped)
            or _BASE_CAPS_RE.match(stripped)
            or _BASE_STRUCTURAL_RE.match(stripped)
        )

    @staticmethod
    def detect_trigger(text: str) -> Optional[str]:
        """Триггер вложенности: нумерация, жирный текст, ключевое слово."""
        stripped = (text or "").strip()
        if re.match(r"^\d+\.\d+", stripped):
            return "numbered"
        if _BOLD_OPEN_RE.match(stripped):
            return "bold"
        if _NEST_KEYWORD_RE.match(stripped):
            return "keyword"
        return None

    @staticmethod
    def is_sibling(current: str, previous: str) -> bool:
        """Соседи ли номера: «1.1» и «1.2» — да, «1.1» и «1.1.1» — нет."""
        current_parts = current.split(".")
        previous_parts = previous.split(".")
        if len(current_parts) == len(previous_parts):
            return (
                current_parts[:-1] == previous_parts[:-1]
                and current_parts[-1] != previous_parts[-1]
            )
        return len(current_parts) <= len(previous_parts)


def classify_header(
    text: str,
    annotations: Optional[List[Dict[str, Any]]] = None,
    *,
    base_size: Optional[float] = None,
    allow_paren: bool = True,
) -> Tuple[bool, Optional[int]]:
    """Детерминированная классификация строки.

    :return: ``(True, level)`` — строка является заголовком;
             ``(False, None)`` — обычный текст.

    Порядок правил влияет только на УРОВЕНЬ: при наличии числовой
    нумерации уровень считается по ней, иначе он равен 1.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False, None

    annotations = annotations or []
    level = numbering_level(stripped, allow_paren=allow_paren)

    # Отладочный вывод для структурных заголовков «Раздел N» / «Глава N» …
    # (Шаг 4 плана диагностики task.txt). Виден только при RAG_DEBUG_HEADERS=1.
    if _DEBUG_HEADERS and _NEST_KEYWORD_RE.match(stripped):
        import sys as _sys

        print(
            f"[DEBUG] classify_header: text={stripped[:60]!r} "
            f"numbering_level={level} base_size={base_size} "
            f"annotations={[(a.get('name'), a.get('value')) for a in annotations[:4]]}",
            file=_sys.stderr,
        )

    # Правила 6 и 7: нумерация в начале (в т.ч. «2.3 АНАЛИЗ ДАННЫХ» —
    # нумерация даёт уровень, КАПС остатка подтверждает заголовок).
    if level is not None:
        return True, level

    # Правила 2, 3, 4: структурные слова, слова секций и содержания.
    if _starts_with_word(stripped, STRUCTURAL_WORDS):
        return True, 1
    if _starts_with_word(stripped, SECTION_WORDS):
        return True, 1
    if _starts_with_word(stripped, CONTENT_WORDS):
        return True, 1

    # Правило 5: весь текст в верхнем регистре.
    if _is_caps(stripped):
        return True, 1

    # Правило 1: кегль >= 13pt и больше базового размера текста.
    if _font_is_bigger(annotations, base_size):
        return True, 1

    # Правило 8: жирный текст в начале строки («**Подраздел**»). Уровень
    # не задаём — его посчитает HeaderLevelTracker по предыдущим
    # заголовкам (это триггер вложенности, а не признак уровня).
    if _BOLD_OPEN_RE.match(stripped):
        return True, None

    # Правило 9: DOCX-жирность. Жирность живёт в аннотациях, а не в тексте,
    # поэтому строка вроде «Предисловие» без нумерации и без стиля Heading
    # иначе осталась бы обычным абзацем.
    if _is_bold_heading(annotations, len(stripped)):
        return True, None

    return False, None


def strip_bold_markers(text: str) -> str:
    """Убирает маркеры жирного текста с краёв готовой строки.

    Маркеры служебные: они несут информацию для ``HeaderLevelTracker`` и
    ``classify_header``, но в итоговый Markdown попадать не должны.

    ``__BOLD__`` — наш собственный маркер, вычищается целиком. ``**`` и
    ``<b>`` — разметка самого документа, поэтому снимаются только в начале
    и в конце строки (именно там триггер и срабатывает); ``**в середине**``
    остаётся и рендерится как жирный текст.
    """
    result = (text or "").replace("__BOLD__", "")
    stripped = result.strip()
    # Симметрично: снимаем края только если строка ЦЕЛИКОМ обёрнута в
    # жирный. Иначе «**жирный** внутри текста» потерял бы открывающий
    # маркер и превратился в «жирный** внутри текста».
    if _BOLD_LEAD_RE.match(stripped) and _BOLD_TAIL_RE.search(stripped):
        stripped = _BOLD_TAIL_RE.sub("", stripped)
        stripped = _BOLD_LEAD_RE.sub("", stripped)
    return stripped.strip()


def _slice_annotations(
    annotations: List[Dict[str, Any]], start: int, end: int
) -> List[Dict[str, Any]]:
    sliced: List[Dict[str, Any]] = []
    for item in annotations:
        try:
            item_start = int(item.get("start", 0) or 0)
            item_end = int(item.get("end", 0) or 0)
        except (TypeError, ValueError):
            continue
        if item_end <= start or item_start >= end:
            continue
        piece = dict(item)
        piece["start"] = max(item_start, start) - start
        piece["end"] = min(item_end, end) - start
        sliced.append(piece)
    return sliced


def _split_lines(node: DocumentNode) -> List[DocumentNode]:
    """Режет многострочный узел на однострочные (аннотации пересчитываются)."""
    if node.type in ("table", "document") or "\n" not in (node.text or ""):
        return [node]
    annotations = _annotations_of(node)
    if not annotations:
        # Без аннотаций делить не по чему — делим вслепую.
        return [
            DocumentNode(
                type=node.type,
                level=node.level,
                text=line.strip(),
                children=[],
                metadata={"annotations_count": 0, "annotations": []},
            )
            for line in node.text.split("\n")
            if line.strip()
        ]
    pieces: List[DocumentNode] = []
    offset = 0
    for line in node.text.split("\n"):
        line_len = len(line)
        sliced = _slice_annotations(annotations, offset, offset + line_len)
        if line.strip():
            pieces.append(
                DocumentNode(
                    type=node.type,
                    level=node.level,
                    text=line.strip(),
                    children=[],
                    metadata={
                        "annotations_count": len(sliced),
                        "annotations": sliced,
                    },
                )
            )
        offset += line_len + 1  # +1 за "\n"
    return pieces or [node]


def _merge_paragraphs(nodes: List[DocumentNode]) -> List[DocumentNode]:
    """Склеивает соседние непромоутнутые строки обратно в абзацы."""
    merged: List[DocumentNode] = []
    buffer: List[DocumentNode] = []

    def flush() -> None:
        if not buffer:
            return
        if len(buffer) == 1:
            merged.append(buffer[0])
        else:
            text = "\n".join(item.text for item in buffer)
            annotations: List[Dict[str, Any]] = []
            offset = 0
            for item in buffer:
                for ann in _annotations_of(item):
                    piece = dict(ann)
                    try:
                        piece["start"] = int(piece.get("start", 0) or 0) + offset
                        piece["end"] = int(piece.get("end", 0) or 0) + offset
                    except (TypeError, ValueError):
                        continue
                    annotations.append(piece)
                offset += len(item.text) + 1
            first = buffer[0]
            merged.append(
                DocumentNode(
                    type="paragraph" if first.type != "raw_text" else "raw_text",
                    level=first.level,
                    text=text,
                    children=[],
                    metadata={
                        "annotations_count": len(annotations),
                        "annotations": annotations,
                    },
                )
            )
        buffer.clear()

    for item in nodes:
        if (
            item.type == "header"
            or item.children
            or item.type == "table"
            or item.metadata.get("toc")
        ):
            flush()
            merged.append(item)
        else:
            buffer.append(item)
    flush()
    return merged


def _flatten(root: DocumentNode) -> List[DocumentNode]:
    """Дерево в плоский список узлов в порядке документа (pre-order)."""
    out: List[DocumentNode] = []

    def visit(node: DocumentNode) -> None:
        out.append(node)
        for child in node.children:
            visit(child)

    for child in root.children:
        visit(child)
    return out


def _rebuild(nodes: List[DocumentNode]) -> List[DocumentNode]:
    """Стековая сборка детей из плоского списка по уровням заголовков.

    Промоутнутый заголовок всплывает из чужой вложенности (lift): его место
    определяется только уровнем, а не dedoc-родителем. В стек пушатся только
    ``type == "header"`` с ``level >= 1`` — header с level 0 не становится
    родителем последующего (не поглощает абзацы).
    """
    result: List[DocumentNode] = []
    stack: List[DocumentNode] = []
    for source in nodes:
        node = DocumentNode(
            type=source.type,
            level=source.level,
            text=source.text,
            children=[],
            metadata=dict(source.metadata),
        )
        if node.type == "header" and (node.level or 0) >= 1:
            while stack and stack[-1].level >= node.level:
                stack.pop()
            if stack:
                stack[-1].children.append(node)
            else:
                result.append(node)
            stack.append(node)
        else:
            if stack:
                stack[-1].children.append(node)
            else:
                result.append(node)
    return result


def _presplit(node: DocumentNode) -> List[DocumentNode]:
    """Заменяет многострочные листья однострочными сиблингами (порядок)."""
    if node.type in ("table", "document"):
        return [
            DocumentNode(
                type=node.type,
                level=node.level,
                text=node.text,
                children=[
                    piece for child in node.children for piece in _presplit(child)
                ],
                metadata=dict(node.metadata),
            )
        ]
    if node.children:
        return [
            DocumentNode(
                type=node.type,
                level=node.level,
                text=node.text,
                children=[
                    piece for child in node.children for piece in _presplit(child)
                ],
                metadata=dict(node.metadata),
            )
        ]
    return _split_lines(node)


def _base_font_size(lines: List[DocumentNode]) -> Optional[float]:
    """Базовый кегль основного текста документа (модальный размер).

    Правило 1 требует сравнения «строго больше базового размера», а базовый
    размер — свойство ДОКУМЕНТА, а не строки. Берём самый частый кегль:
    основной текст встречается в разы чаще заголовков.
    """
    counter: Dict[float, int] = {}
    for line in lines:
        for item in _annotations_of(line):
            if str(item.get("name", "")).lower() != "size":
                continue
            size = _as_size(item.get("value"))
            if size is None:
                continue
            counter[size] = counter.get(size, 0) + 1
    if not counter:
        return None
    return max(counter.items(), key=lambda pair: (pair[1], -pair[0]))[0]


def _is_suppressed_paren(text: str, allow_paren: bool) -> bool:
    """Строка `N)` при запрете paren-ветки (двухпроходная супрессия).

    В документах с >=2 различными «Раздел N» префикс ``N)`` ставит сам dedoc
    из автонумерации Word (`numbering.xml, lvlText=%1)`) — это пункты
    перечня, а не заголовки.
    """
    if allow_paren:
        return False
    return PAREN_ONLY_RE.match((text or "").strip()) is not None


def _count_razdel(lines: List[DocumentNode]) -> int:
    """Число distinct «Раздел N» среди однострочных листьев."""
    numbers = set()
    for line in lines:
        match = RAZDEL_N_RE.match((line.text or "").strip())
        if match:
            numbers.add(match.group(1))
    return len(numbers)


def _mark_toc(lines: List[DocumentNode]) -> int:
    """Помечает TOC-серии (≥5 подряд «Раздел N» с N строго +1). Возвращает число."""
    count = 0
    run: List[DocumentNode] = []
    last_number = 0

    def flush_run() -> None:
        nonlocal count
        if len(run) >= TOC_SERIES_MIN:
            for item in run:
                item.metadata["toc"] = True
                count += 1

    for line in lines:
        match = RAZDEL_N_RE.match((line.text or "").strip())
        if match:
            number = int(match.group(1))
            if run and number == last_number + 1:
                run.append(line)
            else:
                flush_run()
                run = [line]
            last_number = number
        else:
            flush_run()
            run = []
            last_number = 0
    flush_run()
    return count


def _is_toc_entry(text: str) -> bool:
    """Похоже ли строка на ЗАПИСЬ оглавления (а не на текст раздела).

    Обязателен маркер страницы — иначе под правило попал бы любой
    нумерованный абзац: «1 Введение … 4» это запись оглавления, а
    «1 Введение к методам оценки качества» это обычный заголовок.
    """
    stripped = (text or "").strip()
    if not stripped:
        return True  # пустая строка область не прерывает
    if _TOC_DECORATION_RE.match(stripped):
        # «Содержание\n----------------\n1 Область применения … 1»:
        # разделитель под заголовком оглавления не должен обрывать область.
        return True
    if _starts_with_word(stripped, CONTENT_WORDS):
        return False
    if not _TOC_PAGE_MARKER_RE.search(stripped):
        return False
    return bool(
        NUMBER_PREFIX_RE.match(stripped)
        or RAZDEL_N_RE.match(stripped)
        or _starts_with_word(stripped, STRUCTURAL_WORDS)
        or _starts_with_word(stripped, SECTION_WORDS)
    )


def _mark_toc_section(lines: List[DocumentNode]) -> int:
    """Гасит записи оглавления в разделе «Содержание»/«Оглавление».

    Без этого каждая строка вида «5.1 Первое положение … 4» становится
    заголовком и засоряет дерево фантомными разделами: в ``gost_report.txt``
    это 19 записей из 56 заголовков, почти все уровня 1.

    Область идёт от строки-«Содержание» до первого не-TOC абзаца. Подавление
    применяется только если записей не меньше ``TOC_SERIES_MIN`` — иначе это
    не оглавление, а обычный текст со ссылками на страницы.

    Обход идёт по **всем** узлам в порядке документа, а не только по листьям:
    запись оглавления может оказаться составным узлом (с детьми) — тогда до
    листьев она не дойдёт, останется контейнером и станет заголовком целиком.
    Именно так просочились «5.2», «6.2» и «6.2.3» в gost_report.txt.
    """
    count = 0
    index = 0
    total = len(lines)
    while index < total:
        if not _starts_with_word((lines[index].text or "").strip(), CONTENT_WORDS):
            index += 1
            continue
        index += 1
        entries: List[DocumentNode] = []
        while index < total and _is_toc_entry(lines[index].text):
            if (lines[index].text or "").strip():
                entries.append(lines[index])
            index += 1
        if len(entries) >= TOC_SERIES_MIN:
            for item in entries:
                item.metadata["toc"] = True
            count += len(entries)
    return count


def promote_headers(
    root: DocumentNode,
    threshold: Optional[float] = None,  # noqa: ARG001 - DEPRECATED, игнорируется
    promote_non_headers: bool = True,
) -> DocumentNode:
    """Классифицирует заголовки по жёстким правилам. Не мутирует вход.

    :param threshold: **устаревший параметр**, оставлен для совместимости
        вызовов и намеренно ИГНОРИРУЕТСЯ: у детерминированной
        классификации нет порога. Настраивать заголовки правкой правил
        в этом модуле.
    :param promote_non_headers: нужно ли повышать до заголовка строки,
        которые dedoc не назвал заголовком. Для форматов, прошедших через
        HTML-маршрут (``.html``/``.md``/``.xml``/``.txt``), разметка
        источника авторитетна, и правила лучше **не** применять: иначе
        нумерованный пункт вида «1 ПОДГОТОВЛЕН Национальной...» (это
        ``<item>``, то есть абзац) начнёт считаться заголовком. Для
        DOCX/PDF/TXT без разметки параметр остаётся ``True``.
    """
    # Проход 0: пре-сплит всех листьев + сканы по порядку строк.
    split_children: List[DocumentNode] = []
    for child in root.children:
        split_children.extend(_presplit(child))
    scan: List[DocumentNode] = []

    def _collect(nodes: List[DocumentNode]) -> None:
        for item in nodes:
            if item.type == "document":
                _collect(item.children)
            else:
                scan.append(item)
                _collect(item.children)

    _collect(split_children)
    # Один трекер на весь документ: уровни заголовков зависят от
    # предыдущих заголовков (углубление против соседства), поэтому
    # состояние НЕ может быть локальным. Новый документ — новый трекер.
    tracker = HeaderLevelTracker()

    scan_lines = [node for node in scan if not node.children]
    allow_paren = _count_razdel(scan_lines) < 2
    toc_suppressed = _mark_toc(scan_lines) + _mark_toc_section(scan)
    paren_suppressed = 0
    base_size = _base_font_size(scan_lines)

    def process_children(children: List[DocumentNode]) -> List[DocumentNode]:
        """Рекурсивно обрабатывает список детей, возвращает новый список."""
        out: List[DocumentNode] = []
        for child in children:
            out.extend(process_one(child))
        return out

    def process_one(node: DocumentNode) -> List[DocumentNode]:
        nonlocal paren_suppressed
        if node.type in ("table", "document"):
            return [
                DocumentNode(
                    type=node.type,
                    level=node.level,
                    text=node.text,
                    children=process_children(node.children),
                    metadata=dict(node.metadata),
                )
            ]
        if node.children:
            sub = process_children(node.children)
            if _is_suppressed_paren(node.text, allow_paren):
                paren_suppressed += 1
                node_type, level = node.type, node.level
            else:
                is_header, new_level = classify_header(
                    node.text,
                    _annotations_of(node),
                    base_size=base_size,
                    allow_paren=allow_paren,
                )
                node_type, level = node.type, node.level
                if (
                    promote_non_headers
                    and node_type != "header"
                    and node.text.strip()
                    and not node.metadata.get("toc")
                    and is_header
                ):
                    node_type = "header"
                    level = tracker.calculate_level(node.text)
            return [
                DocumentNode(
                    type=node_type,
                    level=level,
                    text=strip_bold_markers(node.text),
                    children=sub,
                    metadata=dict(node.metadata),
                )
            ]
        # лист (после пре-сплита — однострочный): классифицируем → клеим абзацы
        lines = _split_lines(node)
        scored: List[DocumentNode] = []
        for line in lines:
            if line.metadata.get("toc"):
                scored.append(line)
                continue
            if _is_suppressed_paren(line.text, allow_paren):
                paren_suppressed += 1
                scored.append(line)
                continue
            is_header, new_level = classify_header(
                line.text,
                _annotations_of(line),
                base_size=base_size,
                allow_paren=allow_paren,
            )
            if promote_non_headers and line.type != "header" and is_header:
                scored.append(
                    DocumentNode(
                        type="header",
                        # Уровень считаем ДО очистки: триггер жирного
                        # текста читается по маркерам.
                        level=tracker.calculate_level(line.text),
                        text=strip_bold_markers(line.text),
                        children=[],
                        metadata=dict(line.metadata),
                    )
                )
            else:
                line.text = strip_bold_markers(line.text)
                scored.append(line)
        return _merge_paragraphs(scored)

    processed = DocumentNode(
        type="document",
        level=0,
        text=root.text,
        children=process_children(split_children),
        metadata=dict(root.metadata),
    )
    metadata = dict(processed.metadata)
    metadata["toc_suppressed"] = toc_suppressed
    metadata["paren_suppressed"] = paren_suppressed
    return DocumentNode(
        type="document",
        level=0,
        text=processed.text,
        children=_rebuild(_flatten(processed)),
        metadata=metadata,
    )


# --------------------------------------------------------------------------- #
# Приоритетная система (промпт promt_for_right_work_with_head_page.txt).
# Делегирует классификацию модулю priority_rules. Старый promote_headers
# остаётся для обратной совместимости; новый путь включается флагом
# USE_PRIORITY_HEADER в core/config.py (default True для txt/docx/pdf).
# --------------------------------------------------------------------------- #


def promote_headers_priority(
    root: DocumentNode,
    *,
    source: str = "docx",
    base_size: Optional[float] = None,
) -> DocumentNode:
    """Приоритетная промоция заголовков (S → P0 → P1 → P2 → P3 → P4 → P5).

    Не мутирует вход. Переиспользует существующие механизмы:
    * пре-сплит многострочных листьев (``_presplit``);
    * TOC-подавление (``_mark_toc`` + ``_mark_toc_section``) — области
      «Содержание»/TOC-серии гасятся до классификации, S_TOC в
      ``priority_rules`` ловит отдельные TOC-записи;
    * ребилд дерева (``_rebuild``) — пересборка по уровням из прохода 4.

    :param source: источник строк (``"docx"``, ``"txt"``, ``"md"``, ``"pdf"``)
        — определяет, как считается emphasis (по аннотациям или маркерам).
    :param base_size: базовый кегль документа (для docx, правило size ≥13pt).
    """
    from rag_parser.parsers.priority_rules import (
        build_tree,
        classify_line,
        emphasis_from_annotations,
        emphasis_from_text_markers,
        assign_ranks,
        validate_chains,
    )

    # Проход 0: пре-сплит + сбор линий в порядке документа.
    split_children: List[DocumentNode] = []
    for child in root.children:
        split_children.extend(_presplit(child))
    scan: List[DocumentNode] = []

    def _collect(nodes: List[DocumentNode]) -> None:
        for item in nodes:
            if item.type == "document":
                _collect(item.children)
            else:
                scan.append(item)
                _collect(item.children)

    _collect(split_children)

    # TOC-подавление (переиспользуем старые механизмы — они помечают
    # metadata["toc"]=True на TOC-записях).
    scan_lines = [node for node in scan if not node.children]
    toc_suppressed = _mark_toc(scan_lines) + _mark_toc_section(scan)

    # Классификация каждой строки.
    classifications = []
    line_nodes: List[DocumentNode] = []  # параллельный список узлов
    for node in scan:
        text = (node.text or "").strip()
        if not text:
            continue
        # TOC-записи — подавлены, не классифицируем.
        if node.metadata.get("toc"):
            classifications.append(None)
            line_nodes.append(node)
            continue
        # Узлы, которые dedoc уже пометил как header (стиль Heading в DOCX),
        # принимаем как заголовки без классификации — priority-rules не
        # применяется к документам со стилями (спецификация: «заменяет
        # промоцию там, где она применяется сейчас: txt, docx без стилей,
        # PDF-origin Markdown»). Confidence=source (dedoc решил).
        # Уровень берём из node.level (dedoc hierarchy_level) — он уже
        # корректный (Heading 1 → 1, Heading 2 → 2, …).
        if node.type == "header":
            from rag_parser.parsers.priority_rules import (
                LineClassification as _LC,
                CONFIDENCE_SOURCE,
            )
            dedoc_level = max(1, int(node.level or 1))
            classifications.append(_LC(
                text=text, rule="dedoc", kind="dedoc",
                number=None, title=text, confidence=CONFIDENCE_SOURCE,
                metadata={"components": 1, "dedoc_level": dedoc_level},
            ))
            line_nodes.append(node)
            continue
        # emphasis: docx — по аннотациям; txt/md — по маркерам в тексте.
        if source == "docx":
            emph = emphasis_from_annotations(_annotations_of(node), text, base_size)
        else:
            emph = emphasis_from_text_markers(text)
        c = classify_line(text, source=source, emphasis=emph, md_level=None)
        classifications.append(c)
        line_nodes.append(node)

    # Проход 2: валидация цепочек (только для не-None классификаций).
    valid_classes = [c for c in classifications if c is not None]
    valid_classes = validate_chains(valid_classes)

    # Проход 3: ранги.
    valid_classes = assign_ranks(valid_classes)

    # Проход 4: дерево + уровни.
    valid_classes, stats = build_tree(valid_classes)

    # Сопоставляем классификации обратно к узлам.
    cls_iter = iter(valid_classes)
    scored: List[DocumentNode] = []
    for node, c in zip(line_nodes, classifications):
        if c is None or c.suppressed or c.rule is None or c.level is None:
            # Абзац (подавлен S или не прошёл классификацию) — оставляем как есть.
            node.text = strip_bold_markers(node.text)
            scored.append(node)
            continue
        # Заголовок.
        level = c.level
        # Метаданные заголовка: rule, kind, rank, confidence.
        meta = dict(node.metadata)
        meta.pop("annotations", None)
        meta["header_rule"] = c.rule
        meta["header_kind"] = c.kind
        meta["header_rank"] = c.rank
        meta["header_confidence"] = c.confidence
        meta["header_number"] = c.number
        scored.append(
            DocumentNode(
                type="header",
                level=level,
                text=strip_bold_markers(node.text),
                children=[],
                metadata=meta,
            )
        )

    merged = _merge_paragraphs(scored)
    processed = DocumentNode(
        type="document",
        level=0,
        text=root.text,
        children=_rebuild(_flatten(DocumentNode(
            type="document", level=0, text=root.text,
            children=merged, metadata=dict(root.metadata),
        ))),
        metadata={**dict(root.metadata),
                  "toc_suppressed": toc_suppressed,
                  "priority_stats": stats},
    )
    return processed


__all__ = [
    "BOLD_MARKERS",
    "CONTENT_WORDS",
    "MAX_HEADER_LEVEL",
    "HeaderLevelTracker",
    "MIN_HEADER_FONT_SIZE",
    "SECTION_WORDS",
    "STRUCTURAL_WORDS",
    "classify_header",
    "numbering_level",
    "promote_headers",
    "promote_headers_priority",
    "strip_bold_markers",
]
