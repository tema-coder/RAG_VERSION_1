# Парсер документов с чанкингом (dedoc-only)

Бэкенд-сервис на Python: парсинг 9 форматов в Markdown через **единственный
парсер `DedocParser` (dedoc)** со структурным деревом документа,
breadcrumb-обогащением чанков и эвристикой визуальных заголовков, плюс
Streamlit UI.

Архитектура: модульный монолит. Только CPU, без OCR, без векторной БД,
без аутентификации (MVP), синхронный парсинг.

## Возможности

- 9 форматов через dedoc: `.docx`, `.xlsx`, `.xls`, `.pdf` (только текстовый
  слой), `.html`, `.htm`, `.txt`, `.csv`, `.tsv`. Неподдерживаемые (`.doc`,
  `.odt`, сканы) → `422`.
- Структурное дерево `DocumentNode` в ответе API (урезанное превью для
  отображения) и во вкладке UI.
- Breadcrumb-обогащение: `text_for_embedding = "[Раздел 2 > 2.3 ...] текст"`;
  у таблиц префикс в обоих представлениях.
- Header promotion для документов без стилей Word (жирные/нумерованные
  заголовки) с настраиваемым порогом — слайдер в UI и параметр API.
- Forward-fill объединённых ячеек xlsx/xls.
- Гибридный чанкинг на `chonkie`: таблицы — `TableChunker` (Markdown для LLM +
  естественный язык для эмбеддинга), текст — `token` / `sentence` /
  `semantic` / `recursive`.
- REST API (FastAPI) и Streamlit UI.

## Требования

- Python **3.10–3.11** (собрано и протестировано на 3.10.21; 3.12 не
  использовать — у dedoc нет wheels для его пинов).
- Системная библиотека `libmagic` (для `python-magic`, проверка типа файла).
- Java/tesseract **не нужны**: PDF читается только с текстовым слоем
  (`pdf_with_text_layer=true`), OCR и image-ридеры исключены из конфига
  менеджера (см. «Отклонения»).

## Установка

```bash
python3.10 -m venv .venv
source .venv/bin/activate
# 1. Сначала torch CPU-only (ТЗ: только CPU; с дефолтного индекса pip
#    тянет CUDA-сборку ~1.2 ГБ с nvidia-зависимостями):
pip install "torch==2.14.1+cpu" --index-url https://download.pytorch.org/whl/cpu
# 2. Остальное с фиксацией версий:
pip install -r rag_parser/requirements.txt -c constraints.txt
python -c "from dedoc import DedocManager; print('dedoc OK')"
python -c "import chonkie; print(chonkie.__version__)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# ожидаемое: 2.14.1+cpu False, каталог torch ~750 МБ (CUDA — 1.2+ ГБ)
```

Воспроизводимость: `torch` — транзитивная зависимость (dedoc,
`sentence-transformers`), поэтому в `requirements.txt` он не пиннится (прямой
пин ломает резолвер), а фиксируется через `constraints.txt`
(`pip freeze` минус CUDA-пакеты `nvidia-*`/`triton`/`cuda-*`;
`torch==2.14.1` — без local-тега, сборку `+cpu`/`+cu130` выбирает индекс).
Порядок важен: предустановленный `+cpu` резолвер принимает как выполненное
требование и не переустанавливает (`Requirement already satisfied` в логе).
Без шага 1 constraints воспроизведёт CUDA на любой машине.

Зафиксированные версии (пины продиктованы конфликтами зависимостей dedoc,
см. «Отклонения»): `dedoc==2.8`, `fastapi==0.104.1`, `uvicorn==0.23.2`,
`python-multipart==0.0.6`, `python-docx==0.8.11`, `chonkie==1.3.1`,
`httpx==0.27.2`, `numpy==1.26.4`, `scipy==1.11.2`. Остальное — как было
(`streamlit==1.38.0`, `tiktoken==0.7.0` и т.д.).

## Запуск API

Из корня проекта:

```bash
uvicorn rag_parser.main:app --reload --port 8000
```

Swagger UI: <http://localhost:8000/docs>.

## Запуск UI

```bash
streamlit run rag_parser/frontend/app.py
```

UI работает без запущенного API: парсер и чанкеры вызываются напрямую.
Слайдер «Порог распознавания заголовков» (0.0–8.0, default 4.0) + кнопка
«Перепарсить» — повторный разбор с новым порогом без перезагрузки файла.

## Конфигурация

Переменные окружения (файл `.env` создаётся из `.env.example`):

| Переменная            | По умолчанию | Описание                                    |
|-----------------------|--------------|---------------------------------------------|
| `MAX_FILE_SIZE_MB`    | 50           | Максимальный размер загружаемого файла, МБ |
| `DEFAULT_CHUNK_SIZE`  | 200          | Размер чанка по умолчанию, токенов          |
| `DEFAULT_OVERLAP`     | 50           | Перекрытие чанков по умолчанию, токенов     |
| `DEFAULT_TABLE_CHUNK_SIZE` | 5       | Строк данных на табличный чанк              |
| `SEMANTIC_MODEL_NAME` | deepvk/USER-bge-m3 | Модель эмбеддингов для semantic-чанкинга |
| `SEMANTIC_THRESHOLD`  | 0.5          | Порог semantic-чанкинга                     |
| `HEADER_PROMOTION_THRESHOLD` | 4.0   | Порог промоции визуальных заголовков        |
| `BREADCRUMB_MAX_DEPTH`| 3            | Глубина крошки (берутся последние N уровней)|
| `TREE_TEXT_PREVIEW_LEN` | 30         | Длина превью текста узла дерева             |
| `TEMP_DIR`            | ./temp       | Каталог для временных файлов                |
| `DEDOC_OUTPUT_DIR`    | rag_parser/output/dedoc_results | Каталог для результатов dedoc |
| `LOG_LEVEL`           | INFO         | Уровень логирования                         |
| `DOC_STORE_MAX_ITEMS` | 64           | Сколько результатов хранить в памяти        |
| `SUPPORTED_EXTENSIONS`| .docx,...    | Список поддерживаемых расширений (9)        |

## Поддерживаемые форматы

| Формат | Расширения   | Как обрабатывается                                              |
|--------|--------------|-----------------------------------------------------------------|
| DOCX   | `.docx`      | `DocxReader` dedoc напрямую (`DedocManager.parse`)              |
| XLSX   | `.xlsx`      | `ExcelReader` dedoc напрямую — единственный парсер, без обёрток |
| XLS    | `.xls`       | Конвертация xlrd→openpyxl внутри `DedocParser`, далее как xlsx  |
| PDF    | `.pdf`       | `pdf_with_text_layer=true`; без текстового слоя → 400 «OCR отключён» |
| HTML   | `.html`, `.htm` | `HtmlReader` dedoc напрямую                                  |
| TXT    | `.txt`       | `RawTextReader` dedoc напрямую                                  |
| CSV/TSV| `.csv`, `.tsv` | `CSVReader` dedoc напрямую + явные `delimiter`/`encoding=utf-8` |

`.xlsx` идёт в dedoc без посредников (тест `test_xlsx_goes_through_dedoc_without_conversion`
проверяет, что `DedocManager.parse` вызван, а конвертация не запускается, —
защита от будущего увода xlsx мимо dedoc при рефакторинге).

`.xls` конвертируется потому, что dedoc обрабатывает legacy BIFF только через
LibreOffice: `ExcelConverter.convert` выполняет
`soffice --headless --convert-to xlsx`, который исключён ограничениями.
Без конвертации разбор падает с точной ошибкой:

```
DocumentParseError: dedoc failed to parse the document:
[Errno 2] No such file or directory: 'soffice'
```

Конвертация — внутри единого фасада `DedocParser` (отдельного парсера нет),
только CPU. Объединённые диапазоны переносятся (`xlrd` отдаёт
`sheet.merged_cells` только с `formatting_info=True`, есть в BIFF8; в старых
BIFF5 список может быть пуст — тогда для xls работает эвристический
forward-fill).

Сканированные PDF (без текстового слоя) не поддерживаются — OCR отключён.
PDF-таблицы в CPU-окружении без Java/tesseract извлекаются текстом (без
структурированных MD-таблиц): `tabby`-ридер требует Java, pdfminer-ридер —
tesseract для анализа таблиц. Табличные чанки гарантированы для
docx/xlsx/csv.

## Дерево документа

`POST /api/parse` возвращает `tree: TreeNode` (pydantic-рекурсия:
`type/level/text/children/metadata`):

- типы: `document | header | paragraph | list_item | table | raw_text`;
- заголовки dedoc называются `header` (значение `heading` нормализуется);
- `text` — **урезанное превью**: первые `TREE_TEXT_PREVIEW_LEN` (30) символов
  + `…`, только для отображения во фронтенде;
- в `metadata` только `annotations_count` — аннотации целиком не дублируются
  (плюс служебные счётчики `toc_suppressed`, `paren_suppressed`,
  `orphan_tables`);
- полные тексты живут в `.md` и `raw_dedoc_tree_{doc_id}.json`;
- **чанкинг всегда работает по полным текстам**: Markdown собирается из
  полного дерева, урезание применяется последним шагом перед отдачей в API/UI.

Таблицы стоят в дереве на месте из исходника, а не в конце: dedoc вешает на
абзац перед таблицей `TableAnnotation(name="table", value=<uid таблицы>)`,
парсер вставляет таблицу первым ребёнком якоря (до промоции якорь с ребёнком
не режется — привязка выполняется строго после промоции, `reattach_tables`;
аннотации переживают split/merge со пересчётом оффсетов). Без аннотации
(Excel/CSV, дубли, неизвестный uid) — append в конец + warning и счётчик
`orphan_tables` (поведение самого dedoc: его `json2md` делает так же).

На диске (`DEDOC_OUTPUT_DIR`): `raw_dedoc_tree_{id}.json` (сырой JSON dedoc)
и `parsed_{id}.md`. Скачивание: `?format=md|json|tree` (`tree` — урезанное
дерево).

## Breadcrumb-обогащение

Путь заголовков от корня до родителя узла, разделитель `" > "`, берутся
**последние** `BREADCRUMB_MAX_DEPTH` (3) уровней — ближайшие заголовки несут
больше смысла для эмбеддинга. Если путь короче — берётся весь; без
заголовков — `breadcrumb = None`, обогащение не применяется.

- текстовые чанки: `text_for_embedding = "[Раздел 2 > 2.3 ...] текст"`;
- табличные чанки: `[breadcrumb]` в начале **обоих** представлений (`text` и
  `text_for_embedding`); NL-представление не содержит `|`.

## Header promotion

Документы без стилей Word (жирные/крупные/центрированные «Раздел N.»)
получают заголовки эвристикой до генерации Markdown и чанкинга:

| Признак | Баллы |
|---|---|
| Regex `^(Раздел\s+\d+\.?|\d+(\.\d+)*\.?)(\s|$)` или `^\d+\)` | +4 явный «Раздел N», иначе +3; level из нумерации (см. ниже) |
| `bold=True` | +2 |
| `size > 14` | +1.5 |
| КАПС | +1 |
| Длина < 100 | +1 |
| `alignment=center` | +0.5 |
| Предложение с точкой длиннее 120 | −2 |

Уровень из нумерации: точки считаются только между цифрами («Раздел 2.» → 1,
«2.» → 1, «2.3» → 2, «2.3.1» → 3); ветка `N)` — подчинённый уровень 2.
Отклонение от исходной шкалы (+3): явный «Раздел N» даёт +4 — длинные титулы
секций иначе не добирали балл и их таблицы приписывались предыдущему разделу
(проверено на годовом отчёте).

Порог — `HEADER_PROMOTION_THRESHOLD` (4.0), переопределяется параметром API
`header_threshold` и слайдером UI. Многострочные узлы предварительно режутся
на строки (аннотации пересчитываются по диапазонам), затем абзацы склеиваются
обратно. Финал — глобальный ребилд: дерево сплющивается в порядок документа
и пересобирается стеком по уровням, поэтому промоутнутый заголовок всплывает
из чужой dedoc-вложенности (lift) вместо погребения в ней.

Два doc-level правила (считаются до скоринга, в `tree.metadata` — счётчики
`toc_suppressed` / `paren_suppressed`):
- **Супрессия `N)`**: при ≥2 distinct «Раздел N» в документе строки вида
  `N)` не промоутятся вовсе (это пункты перечня, а не заголовки; префикс
  `N)` ставит сам dedoc из автонумерации Word — `numbering.xml, lvlText=%1)`).
  Без «Раздел N» поведение прежнее. Абсолютна: не зависит от порога.
- **TOC-серии**: ≥5 подряд идущих «Раздел N» с N строго +1 (оглавление)
  остаются paragraph поштучно, иначе абзацы между TOC и телом получали бы
  ложный breadcrumb от последней строки оглавления.

## Типы чанкинга

| Тип        | Текстовая стратегия (`chonkie`) | Параметры |
|------------|---------------------------------|-----------|
| `none`     | —                               | `chunks = null` |
| `fixed`    | `TokenChunker` (`cl100k_base`)  | `chunk_size` (200), `overlap` (50) |
| `sentence` | `SentenceChunker`               | `chunk_size` (200) |
| `semantic` | `SemanticChunker`               | `chunk_size` (200), `threshold` (0.5) |
| `recursive`| `RecursiveChunker`              | `chunk_size` (200), `overlap` (20) |

`chunk_size` — 100…2000, `overlap` — 0…500 (< `chunk_size`), `table_chunk_size`
— 1…50, `threshold` — строго (0, 1), `header_threshold` — 0…8.
Семантический чанкер тянет модель с Hugging Face (~2 ГБ), нужен интернет.

## Примеры использования API

```bash
curl -X POST http://localhost:8000/api/parse \
  -F "file=@report.pdf" \
  -F "chunking_type=none"

curl -X POST http://localhost:8000/api/parse \
  -F "file=@complex_structured.docx" \
  -F "chunking_type=recursive" \
  -F "chunk_size=500" \
  -F "overlap=50" \
  -F "header_threshold=2.0"

curl -OJ "http://localhost:8000/api/parse/<document_id>/download?format=tree"
```

### HTTP-коды

| Код  | Случай                                                            |
|------|-------------------------------------------------------------------|
| 200  | Успех                                                             |
| 400  | Битый файл, неверные параметры чанкинга                           |
| 404  | `document_id` не найден; удалённый `/api/parse_dedoc`             |
| 413  | Файл больше `MAX_FILE_SIZE_MB`                                    |
| 422  | Неподдерживаемый формат (`.doc`, `.odt`, `.exe`, …)               |
| 500  | Непредвиденная ошибка сервера                                     |

## Тесты

```bash
python -m pytest
python -m pytest --cov=rag_parser.parsers --cov=rag_parser.chunkers --cov-fail-under=80
```

Покрытие: parsers ~85%, chunkers ~96%. Фикстуры: `.docx`/`.xlsx` — через
python-docx/openpyxl, `.xls` — предсгенерирован (`test_data/sample.xls`,
генератор — xlwt), PDF — собственный генератор без зависимостей.
`test_data/merged_cells.xlsx` — merged-диапазоны (горизонтальные, вертикаль,
2D-блок), `complex_structured.docx` — эталонный документ (перегенерируется
скриптом при отсутствии).

## Структура проекта

```
rag_parser/
├── api/            # FastAPI: схемы (TreeNode), фабрика DedocParser, роуты
├── parsers/        # dedoc_parser (единственный) + dedoc_tree + merged_cells + header_promoter
├── chunkers/       # token, sentence, semantic, recursive, table (chonkie) + hybrid + breadcrumb
├── frontend/       # Streamlit UI (дерево, слайдер порога, скачивание)
├── core/           # Настройки и интерфейсы (DocumentNode, Chunk.breadcrumb, ParseResult.tree)
├── scripts/        # Генерация тестового DOCX
├── test_data/      # complex_structured.docx, merged_cells.xlsx, sample.xls
├── output/         # Результаты dedoc (в .gitignore)
├── tests/          # test_dedoc_tree, test_merged_cells, test_header_promoter, test_breadcrumb, ...
├── main.py
├── requirements.txt
└── .env.example
```

## Миграция с markitdown: что удалено и почему

Удалены `parsers/markitdown_parser.py`, `parsers/xlsx_parser.py` (calamine),
`parsers/csv_parser.py`, их тесты и фабрики, эндпоинт `/api/parse_dedoc`
(остался один `/api/parse`), зависимости `markitdown[all]` и
`python-calamine`. Причины: единый структурный парсер (дерево + таблицы +
иерархия) вместо трёх «плоских» конвертеров; dedoc отдаёт merged-диапазоны и
иерархию, которых у конвертеров не было. В коде не осталось упоминаний
markitdown/calamine (этот раздел — единственное исключение).

Forward-fill merged cells перенесён из старого xlsx-парсера и усилен до точной
заливки: `DedocParser` читает реальные диапазоны книги (`openpyxl`,
таблица `i` ↔ лист `i`, размерность сверяется — при несовпадении потабличный
fallback на эвристику с warning) и копирует значение левой-верхней ячейки во
все клетки диапазона независимо от направления (вправо/вниз/2D).
Эвристика «слева, затем сверху» осталась только fallback без доступа к файлу;
именно она была неточна на вертикальных мёрджах с непустым левым соседом.

## Отклонения от ТЗ

1. **Даунгрейды ради установки dedoc** (согласованы): `fastapi 0.115.0 →
   0.104.1` (`starlette>=0.37` vs `starlette<=0.27` у dedoc 2.8 — единственной
   версии на PyPI, новее нет), `uvicorn 0.30.6 → 0.23.2`,
   `python-multipart 0.0.9 → 0.0.6`, `python-docx → 0.8.11` (только фикстуры),
   `chonkie 1.7.0 → 1.3.1` (`numpy>=2` vs `scipy<=1.11.2 → numpy<1.28`),
   `httpx 0.28.1 → 0.27.2` (TestClient starlette 0.27). Обёртки чанкеров
   адаптированы: в 1.3.x первый аргумент называется
   `tokenizer_or_token_counter` (передаём позиционно),
   `TableChunker(tokenizer="row")` в 1.3.x нет — работает штатный fallback с
   ручной группировкой строк. Поведение стратегий и тестов сохранено.
2. **`DedocManager` с урезанным `manager_config`** (а не дефолтным): дефолт
   поднимает `PdfImageReader` (torchvision, OCR) — запрещено ограничениями.
   Включены только Docx/Excel/Txt/Csv/Html-ридеры и PDF-ридеры текстового слоя;
   выбор ридера по расширению — внутри `ReaderComposition`.
3. **CSV/TSV читаются dedoc нативно** (`CSVReader`), конвертация в txt не
   понадобилась. Но: `delimiter` (автодетект/таб) и `encoding=utf-8`
   передаются явно — детектор dedoc (charset_normalizer) на короткой кириллице
   ошибается (`utf_16_be`, кракозябры).
4. **PDF-таблицы без Java/tesseract — текстом** (см. «Форматы»).
5. **`ParseResult.tree` внутри сервиса — полное дерево**, превью (30 символов)
   строит слой API/UI последним шагом — иначе чанкинг работал бы по
   обрезанному тексту.
6. Тесты API ходят через `fastapi.testclient.TestClient` (как раньше);
   `performance`-тест PDF ослаблен до 60 c (реальный парсинг dedoc тяжелее
   markitdown).
7. `xlwt` (без зависимостей) добавлен только для генерации `.xls`-фикстур.
8. Хранилище результатов — в памяти (`DocumentStore`, LRU-64), ошибки
   валидации запроса — 400, неподдерживаемые форматы — 422 (единый код по ТЗ).
9. **Ребилд иерархии после промоции** (вне ТЗ, по итогам e2e на реальном
   годовом отчёте): dedoc вкладывает контент в поддерево list-узлов
   (автонумерация Word → префиксы `N)`), а стековая сборка без ребилда
   никогда не выносит промоутнутый заголовок из чужой вложенности —
   «Раздел 2» погребается под пунктом «3)». Поэтому дерево сплющивается
   и пересобирается по уровням (lift), `N)`-пункты подавляются в
   «Раздел N»-документах, TOC-серии (≥5) не промоутятся. Проверено на
   годовом отчёте: 24 заголовка, 0 крошек с `N)`, 0 крошек с двумя
   «Раздел N».
10. **Привязка таблиц по `TableAnnotation`** (в dedoc 2.8 нет
    `paragraph_node_id` — проверено grep по пакету): таблицы вставляются
    первым ребёнком якоря после промоции; порядок 8/8 сверен с нативным
    `json2md` (у него те же правила: inline по аннотации, остаток в конец).
