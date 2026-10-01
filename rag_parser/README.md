# Парсер документов с чанкингом

Бэкенд-сервис на Python для парсинга документов различных форматов в Markdown
с опциональным чанкингом, плюс минимальный веб-интерфейс на Streamlit для
загрузки и скачивания результатов.

Архитектура: модульный монолит. Только CPU, без OCR, без векторной БД,
без аутентификации (MVP), синхронный парсинг.

## Возможности

- 8 форматов: PDF, DOCX, HTML, TXT/MD (через `markitdown`), XLSX/XLS (собственный парсер на `python-calamine` с поддержкой объединённых ячеек) и CSV (собственный парсер).
- Таблицы из XLSX конвертируются в валидные Markdown-таблицы, листы разделяются заголовками `## Лист: {название}`.
- Гибридный чанкинг на `chonkie`: таблицы всегда чанятся `TableChunker` (два представления — Markdown для LLM и естественный язык для эмбеддинга), текст — на выбор `token` / `sentence` / `semantic`.
- REST API (FastAPI) и Streamlit UI.
- Логирование операций (structlog, уровень INFO+), конфигурация через ENV.

## Требования

- Python 3.10–3.12 (проект собран и протестирован на 3.12).
- Системная библиотека `libmagic` (для `python-magic`, проверка типа файла).

## Установка

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r rag_parser/requirements.txt
```

Проверка зависимостей:

```bash
pip check
```

## Запуск API

Из корня проекта:

```bash
uvicorn rag_parser.main:app --reload --port 8000
```

Из каталога `rag_parser/`:

```bash
uvicorn main:app --reload --port 8000
```

Swagger UI: <http://localhost:8000/docs>, ReDoc: <http://localhost:8000/redoc>.

## Запуск UI

```bash
streamlit run rag_parser/frontend/app.py
```

Или из каталога `rag_parser/`:

```bash
streamlit run frontend/app.py
```

UI работает без запущенного API: парсеры и чанкеры вызываются напрямую.

## Конфигурация

Переменные окружения (файл `.env` создаётся из `.env.example`):

| Переменная            | По умолчанию | Описание                                    |
|-----------------------|--------------|---------------------------------------------|
| `MAX_FILE_SIZE_MB`    | 50           | Максимальный размер загружаемого файла, МБ |
| `DEFAULT_CHUNK_SIZE`  | 200          | Размер чанка по умолчанию, токенов          |
| `DEFAULT_OVERLAP`     | 50           | Перекрытие чанков по умолчанию, токенов     |
| `DEFAULT_TABLE_CHUNK_SIZE` | 5       | Строк данных на табличный чанк              |
| `SEMANTIC_MODEL_NAME` | deepvk/USER-bge-m3 | Модель эмбеддингов для semantic-чанкинга |
| `TEMP_DIR`            | ./temp       | Каталог для временных файлов                |
| `LOG_LEVEL`           | INFO         | Уровень логирования                         |
| `DOC_STORE_MAX_ITEMS` | 64           | Сколько результатов хранить в памяти        |
| `SUPPORTED_EXTENSIONS`| .pdf,...     | Список поддерживаемых расширений            |

## Поддерживаемые форматы

| Формат | Расширения         | Парсер             | Особенности                                                     |
|--------|--------------------|--------------------|-----------------------------------------------------------------|
| PDF    | `.pdf`             | MarkItDownParser   | Текст всех страниц, таблицы — в виде Markdown-таблиц            |
| XLSX   | `.xlsx`, `.xls`    | XlsxParser         | `CalamineWorkbook`: все листы под заголовком `## Лист: {название}`, объединённые ячейки заполняются (forward-fill) |
| DOCX   | `.docx`            | MarkItDownParser   | Структура, заголовки, таблицы                                   |
| HTML   | `.html`, `.htm`    | MarkItDownParser   | Заголовки h1–h6, таблицы, ссылки                                |
| TXT/MD | `.txt`, `.md`      | MarkItDownParser   | Определение кодировки charset-normalizer                         |
| CSV    | `.csv`, `.tsv`     | CsvParser          | Собственный парсер: автодетект разделителя, стабильные таблицы  |

Сканированные PDF (без текстового слоя) не поддерживаются — OCR не используется.
PPTX, медиафайлы и архивы сознательно не поддерживаются: расширения из этого
списка отклоняются с ошибкой 400.

Метаданные от `MarkItDownParser`: `format`, `extension`, `size_bytes`, `lines`,
`characters`, `converter`. У CSV дополнительно: `rows`, `columns`, `delimiter`,
`header`. Значение `pages` в ответе API всегда `null` (см. отклонения).

## Типы чанкинга

Гибридная модель: при любом типе (кроме `none`) документ сначала делится на
блоки по пустым строкам, затем **табличные блоки всегда чанятся
`TableChunker`** (заголовок таблицы прикрепляется к каждому чанку), а
текстовые — выбранной стратегией.

| Тип        | Текстовая стратегия (`chonkie`) | Параметры |
|------------|---------------------------------|-----------|
| `none`     | —                               | Текст возвращается целиком, `chunks = null` |
| `fixed`    | `TokenChunker` (`cl100k_base`)  | `chunk_size` (200), `overlap` (50) |
| `sentence` | `SentenceChunker`               | `chunk_size` (200), предложения не разрываются |
| `semantic` | `SemanticChunker`               | `chunk_size` (200), `threshold` (0.5) |

Табличные чанки (`metadata.chunk_type == "table"`) несут два представления:
`text` — оригинальный Markdown с `|` (для контекста LLM) и
`text_for_embedding` — естественный язык вида `Заголовок: значение, ...`
(для эмбеддинга). Размер табличного чанка — `table_chunk_size` (5) строк
данных. У текстовых чанков `text_for_embedding = null`.

`chunk_size` — 100…2000 токенов, `overlap` — 0…500 (и всегда меньше `chunk_size`),
`table_chunk_size` — 1…50, `threshold` — строго между 0 и 1.

Семантический чанкер использует модель `deepvk/USER-bge-m3` (переопределяется
через `SEMANTIC_MODEL_NAME`, указывается без префикса провайдера).
**Внимание:** при первом запуске модель скачивается с Hugging Face Hub
(~2 ГБ), нужен интернет; работает на CPU. Тесты для semantic-чанкера
помечены `@pytest.mark.slow` и пропускаются без сети.

## Примеры использования API

Проверка состояния:

```bash
curl http://localhost:8000/api/health
```

Список форматов и лимитов:

```bash
curl http://localhost:8000/api/formats
```

Парсинг без чанкинга:

```bash
curl -X POST http://localhost:8000/api/parse \
  -F "file=@report.pdf" \
  -F "chunking_type=none"
```

Парсинг с фиксированным чанкингом:

```bash
curl -X POST http://localhost:8000/api/parse \
  -F "file=@report.pdf" \
  -F "chunking_type=fixed" \
  -F "chunk_size=500" \
  -F "overlap=50"
```

Семантический чанкинг:

```bash
curl -X POST http://localhost:8000/api/parse \
  -F "file=@report.pdf" \
  -F "chunking_type=semantic" \
  -F "chunk_size=800" \
  -F "overlap=100"
```

Скачивание результата (по `document_id` из ответа):

```bash
curl -OJ "http://localhost:8000/api/parse/<document_id>/download?format=md"
curl -OJ "http://localhost:8000/api/parse/<document_id>/download?format=json"
```

Пример ответа:

```json
{
  "document_id": "9a1f...",
  "filename": "report.pdf",
  "format": "pdf",
  "pages": 10,
  "markdown": "## Страница 1\n...",
  "chunks": [
    {
      "index": 0,
      "text": "Текст чанка...",
      "tokens": 480,
      "source": "report.pdf",
      "metadata": {"chunker": "fixed", "start_token": 0, "end_token": 500}
    }
  ],
  "metadata": {"pages": 10, "tables": 1, "tables_detail": []}
}
```

### HTTP-коды

| Код  | Случай                                                            |
|------|-------------------------------------------------------------------|
| 200  | Успех                                                             |
| 400  | Неподдерживаемый формат, битый файл, неверные параметры чанкинга |
| 404  | `document_id` не найден при скачивании                            |
| 413  | Файл больше `MAX_FILE_SIZE_MB`                                    |
| 500  | Непредвиденная ошибка сервера                                     |

## Тесты

```bash
python -m pytest
```

Покрытие парсеров и чанкеров:

```bash
python -m pytest --cov=rag_parser.parsers --cov=rag_parser.chunkers --cov-fail-under=80
```

Тесты сами генерируют тестовые файлы всех форматов во временном каталоге:
`.docx` и `.xlsx` — через python-docx и openpyxl, PDF — небольшим собственный
генератором без внешних зависимостей (встроенный шрифт Helvetica, поэтому
текст в PDF-фикстурах латиницей). Отдельно проверяются битые PDF/DOCX/XLSX,
пустые файлы, архивы и неподдерживаемые расширения.

## Структура проекта

```
rag_parser/
├── api/            # FastAPI: схемы, зависимости (фабрики, валидация, хранилище), роуты
├── parsers/        # MarkItDownParser (pdf, docx, html, txt, md) + XlsxParser + CsvParser
├── chunkers/       # token, sentence, semantic, table (chonkie) + hybrid-оркестратор
├── frontend/       # Streamlit UI
├── core/           # Настройки (pydantic-settings) и интерфейсы (ABC, Chunk, ParseResult)
├── tests/          # Тесты парсеров, чанкеров и API
├── main.py         # Точка входа FastAPI, логирование
├── requirements.txt
└── .env.example
```

## Отклонения от ТЗ

### Миграция чанкеров на `chonkie`

1. **`chonkie==1.7.0` + `sentence-transformers>=3.0` вместо
   `semantic-chunkers`.** Ручные реализации на `tiktoken`/regex/`ConsecutiveChunker`
   удалены (`chunkers/fixed_token.py` удалён, `sentence.py`/`semantic.py`
   переписаны как обёртки над chonkie). Общие tiktoken-хелперы
   (`count_tokens`, `get_encoding`) переехали в `chunkers/base.py`.
2. **Новый `chunkers/table.py`.** `TableChunker(tokenizer="row")` прикрепляет
   заголовок к каждому чанку; предусмотрен fallback с ручной группировкой
   строк, если установленный chonkie не поддерживает `tokenizer="row"`.
   Пустые строки-заглушки `| | |` в начале таблицы вычищаются, после
   promoted-заголовка вставляется свежий разделитель.
3. **Новый `chunkers/hybrid.py`.** `HybridRagChunker` — оркестратор: сплит по
   `\n\n+`, табличные блоки → `TableChunkerRAG`, текстовые → выбранная
   стратегия. Фабрика чанкеров в API возвращает `HybridRagChunker`; тип `NONE`
   по-прежнему отключает чанкинг. Тип чанка — в
   `metadata.chunk_type` (`"table"`/`"text"`), NL-представление — в поле
   `Chunk.text_for_embedding` / `ChunkResponse.text_for_embedding`.
4. **`SemanticChunker.threshold` в chonkie — строго в (0, 1).** API валидирует
   `threshold` как `0 < threshold < 1` (иначе 400); в схеме оставлен
   `Field(0.5, ge=0.0, le=1.0)` для документации диапазона. Параметр chonkie
   называется `threshold` (не `similarity_threshold`), идентификатор модели —
   `sentence-transformers/<имя>`. Чанкер создаётся лениво, при первом `chunk()`.
5. **`httpx==0.28.1`** (было 0.27.2): chonkie требует `httpx>=0.28.1`.
   `DEFAULT_CHUNK_SIZE` изменён 500 → 200 по ТЗ; добавлен
   `DEFAULT_TABLE_CHUNK_SIZE=5`, `SEMANTIC_MODEL_NAME`, `SEMANTIC_THRESHOLD`.

### XLSX на `python-calamine`

6. **`XlsxParser` на `CalamineWorkbook`** (`python-calamine==0.3.1`).
   `.xlsx`/`.xls` убраны из `MarkItDownParser.supported_extensions`, фабрика
   отдаёт их `XlsxParser`. В calamine 0.3.1 нет API merged-диапазонов,
   поэтому forward-fill: пустая ячейка наследует значение слева, затем сверху.
   Горизонтальные мёрджи и строки-заголовки покрыты; вертикальные мёрджи в
   середине таблицы могут заполняться неточно (ограничение задокументировано).

### Миграция на `markitdown` (сохранено)

1. **`markitdown[all]==0.1.8` вместо отдельных библиотек.** Удалены
   `mammoth`, `markdownify`, `pdfplumber`, `pymupdf` (часть вернётся
   транзитивно внутри `markitdown[all]` — это нормально, в `requirements.txt`
   они больше не перечисляются). `python-calamine` возвращён отдельной строкой
   для собственного `XlsxParser`. Экстра `[all]` нужна, иначе при конвертации
   docx/pdf возникает `MissingDependencyException`.
2. **Пин версии `0.1.8`.** Экстра `[all]` тянет azure/speech/magika-пакеты,
   которые меняются между релизами; без пина установка не воспроизводима.
3. **`.xlsm` больше не поддерживается.** Поддерживаемые расширения:
   `.pdf`, `.docx`, `.html`, `.htm`, `.txt`, `.md` (markitdown),
   `.xlsx`, `.xls` (собственный `XlsxParser`), `.csv`, `.tsv` (собственный `CsvParser`).
4. **Потеряны метаданные `pages`, `tables`, `sheets`, `headings`.** markitdown
   отдаёт только текст, поэтому в ответе API `pages` всегда `null`, а таблицы
   присутствуют в Markdown, но без координат и номеров.
5. **Разделители страниц `## Страница N` и метки `[ТАБЛИЦА N]` убраны** —
   это разметка предыдущего парсера. Символы перевода страниц (`\f`)
   нормализуются в пустые строки.
6. **Добавлена проверка сигнатуры файла.** При битом `.docx`/`.pdf`
   markitdown молча переключается на конвертер простого текста и возвращает
   «мусор» из бинарника как Markdown. `MarkItDownParser` проверяет заголовок
   файла (`PK`, `%PDF`) и отклоняет файл с понятной ошибкой 400.
7. **PyMuPDF убран и из тестовых фикстур.** PDF для тестов генерируется
   небольшим собственным генератором (~60 строк) без внешних зависимостей.

### Прочие отклонения (сохранены с прошлых версий)

7. **Добавлены `tqdm`, `pytest-cov`, `openpyxl`, `python-docx` и ограничение
   `pydantic<3`.** `pytest-cov` нужен для проверки покрытия, `openpyxl`/`python-docx` —
   для генерации тестовых файлов.
8. **`rag_parser/__init__.py` добавлен** (в дереве ТЗ отсутствует) — необходим для
   импортов вида `rag_parser.api.routes`.
9. **Хранилище результатов — в памяти** (`DocumentStore`, LRU на 64 документа, без БД),
   чтобы работал `GET /api/parse/{document_id}/download`. Результаты не переживают
   перезапуск процесса.
10. **Ошибки валидации запроса возвращают 400, а не 422** (единый код «неверные
    параметры» по ТЗ).
