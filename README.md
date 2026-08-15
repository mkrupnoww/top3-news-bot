# TOP 3 NEWS — Telegram-бот киноновостей

TOP 3 NEWS — production-проект для автоматического поиска, анализа, отбора, подготовки и публикации трёх главных киноновостей дня в Telegram.

Система строится как ежедневный restart-safe pipeline:

```text
RSS-источники
        ↓
сбор новостей по расписанию
        ↓
PostgreSQL
        ↓
OpenAI ranking
        ↓
сохранённые ranking combinations
        ↓
TOP-3
        ↓
генерация текста
        ↓
автоматический self-review
        ↓
генерация общей PNG-иллюстрации
        ↓
каскад восстановления при image moderation
        ↓
контроль готовности публикации
        ↓
Telegram
```

Основная цель проекта — **полностью автоматическая ежедневная публикация без участия человека**.

Текущий production-этап уже автоматизирует сбор, ranking, генерацию текста, self-review, генерацию изображения, восстановление после части отказов Image API и доставку готового поста на review. Ручное подтверждение пока сохраняется как переходный safety gate перед окончательным переходом к full-auto режиму.

---

## Целевая архитектура

Целевой production-сценарий:

```text
Почасовой RSS collector
        ↓
PostgreSQL
        ↓
Ежедневный systemd timer
        ↓
24-часовое окно кандидатов
        ↓
исключение уже опубликованных news_id
        ↓
OpenAI event ranking
        ↓
оценка кандидатов и ranking combinations
        ↓
выбор winner TOP-3
        ↓
генерация Telegram-текста
        ↓
автоматический OpenAI self-review
        ↓
финальный текст
        ↓
генерация общей PNG-иллюстрации
        ↓
многоступенчатое восстановление при moderation_blocked
        ↓
автоматическая проверка готовности
        ↓
публикация native photo + caption
        ↓
фиксация Telegram message_id и результата публикации
```

В конечной версии человек не должен участвовать в штатном ежедневном цикле.

Ручной review остаётся только как временный переходный режим, диагностический инструмент и аварийная административная возможность.

---

## Текущий production-пайплайн

На текущем этапе система уже поддерживает:

* почасовой сбор новостей из RSS;
* сохранение исходных материалов в PostgreSQL;
* 24-часовое окно кандидатов;
* исключение уже опубликованных `news_id`;
* OpenAI event ranking;
* сохранение не только winner TOP-3, но и множества ranking combinations;
* выбор трёх наиболее сильных и разнообразных новостей;
* генерацию Telegram-поста через OpenAI API;
* отдельный автоматический self-review текста;
* лимит Telegram photo caption;
* генерацию единой вертикальной PNG-иллюстрации;
* restart-safe checkpoints ежедневного workflow;
* idempotency для ranking, text generation, image generation и publication;
* защиту от повторных платных API-вызовов после рестарта;
* многоступенчатый recovery после `moderation_blocked`;
* замену проблемной TOP-3 combination без повторного ranking;
* сохранение полной истории replacement attempts;
* перевод старых batch/post в `superseded`;
* native Telegram photo + caption;
* ручной review через Telegram-кнопки;
* отдельный publisher для approved постов;
* ежедневный запуск production workflow через `systemd`.

---

## Ежедневное расписание

Основной daily workflow запускается через:

```text
top3-news-daily.timer
```

Текущее расписание:

```text
07:30 UTC
```

Timer запускает production workflow один раз в день.

Отдельно работают:

```text
top3-news-collector.timer
top3-news-cleanup.timer
```

RSS collector выполняет сбор новостей независимо от ежедневного production workflow.

---

# Отбор TOP-3

## Индивидуальная оценка новости

Каждая новость анализируется OpenAI ranking pipeline по версии зафиксированной формулы и структурированных параметров.

Базовая индивидуальная формула:

```text
B_i = 0.20 × F_i
    + 0.30 × M_i
    + 0.20 × R_i
    + 0.15 × (H_i × Q_i)
```

Где:

* `F` — свежесть и актуальность;
* `M` — масштаб события и потенциальный охват;
* `R` — общественный резонанс и вовлечённость;
* `H` — необычность, конфликтность или другой цепляющий фактор;
* `Q` — подтверждённость информации и качество источников.

В production ranking используется расширенный набор содержательных параметров и версионируемая формула.

Для каждого ranking run фиксируются:

* версия формулы;
* версия evaluator;
* версия prompt;
* модель;
* временное окно;
* список кандидатов;
* индивидуальные оценки;
* eligibility;
* причины исключения;
* итоговый ranking;
* ranking combinations;
* winner combination;
* explanation и структурированные данные OpenAI.

---

## Ranking combinations

Ranking pipeline сохраняет не только одну итоговую тройку.

После оценки кандидатов формируется набор допустимых сочетаний TOP-3:

```text
combination #1  ← winner
combination #2
combination #3
combination #4
...
```

Каждая combination имеет собственный `combination_rank`.

Это позволяет не запускать ranking заново, если позже оказывается, что конкретная тройка создаёт устойчивую проблему на стадии генерации изображения.

Replacement selector:

1. исключает уже использованные combinations;
2. исключает точное повторение текущей тройки;
3. максимизирует overlap с текущим TOP-3;
4. среди одинакового overlap выбирает combination с лучшим `combination_rank`.

Например:

```text
winner:
{A, B, C}

следующая combination:
{A, D, C}
```

В таком случае заменяется только одна новость.

---

# Генерация текста

Для выбранной combination создаётся отдельный publication batch.

Text pipeline:

```text
TOP-3
        ↓
primary generation
        ↓
automatic self-review
        ↓
final Telegram post
        ↓
generated_post
```

Основной generation и self-review являются двумя отдельными OpenAI-вызовами.

Self-review получает уже подготовленный текст и проверяет:

* соответствие исходным материалам;
* неподтверждённые утверждения;
* имена;
* даты;
* названия;
* числовые данные;
* ссылки;
* качество заголовков;
* структуру текста;
* Telegram-форматирование;
* пригодность к публикации.

В PostgreSQL сохраняется только финальный пост, а telemetry и token usage двух OpenAI-вызовов агрегируются.

---

## Контракт Telegram-текста

Для публикации используется native Telegram photo + caption.

Текущий контракт:

```text
Telegram hard limit: 1024 символа
Project max:         1000 символов
Target:              850–950 символов
```

Текст формируется с учётом ограничений Telegram до стадии публикации.

---

# Генерация изображения

## Формат

Для трёх новостей создаётся **одно вертикальное PNG-изображение**.

Базовая композиция:

```text
┌─────────────────────────────┐
│           News 1            │
├─────────────────────────────┤
│           News 2            │
├─────────────────────────────┤
│           News 3            │
└─────────────────────────────┘
```

Правила:

* три основные горизонтальные зоны;
* примерно одинаковая высота;
* чёткие разделители;
* единый редакционный стиль;
* без нумерации;
* без общего заголовка TOP-3;
* без логотипов;
* без копирования официальных постеров и кадров;
* без смешивания элементов разных новостей;
* PNG используется как native Telegram photo.

---

# Каскад восстановления после image moderation

Одна из ключевых частей проекта — **production recovery при отказе Image API по `moderation_blocked`**.

Отказ генерации одного изображения не должен автоматически приводить к потере ежедневной публикации.

Текущая архитектура:

```text
TOP-3 combination
        ↓
NORMAL IMAGE
        ↓
success ───────────────────────→ publish/review
        │
        └─ moderation_blocked
                ↓
SEMANTIC SAFE FALLBACK #1
                ↓
        success ───────────────→ publish/review
                │
                └─ moderation_blocked
                        ↓
SEMANTIC SAFE FALLBACK #2
                        ↓
                success ───────→ publish/review
                        │
                        └─ moderation_blocked
                                ↓
выбрать следующую ranking combination
                                ↓
старый batch/post → superseded
                                ↓
новая TOP-3
                                ↓
новый текст
                                ↓
новый image budget
                                ↓
повторить image cascade
```

---

## Normal image mode

Первый запрос использует основной визуальный prompt.

Он максимально точно передаёт:

* конкретную новость;
* фильм;
* жанр;
* событие;
* известных публичных людей, когда это допустимо;
* персонажей, когда это допустимо;
* киноиндустриальный контекст.

При этом prompt требует оригинальную редакционную композицию и запрещает буквальное копирование защищённых материалов.

---

## Semantic moderation-safe fallback

После доказанного `moderation_blocked` используется отдельный fallback prompt с отдельной `prompt_version`.

Fallback не получает исходные точные идентификаторы.

Исходные `title` и `summary` анализируются локально, после чего Image API получает только безопасный структурированный `semantic_visual_brief`.

Сохраняются безопасные смысловые признаки:

```text
news_type
genre
setting
action
atmosphere
visual_motifs
palette
people
avoid
```

При этом из fallback-запроса удаляются:

* точные имена актёров;
* имена режиссёров;
* имена других публичных людей;
* названия конкретных персонажей;
* названия франшиз;
* названия фильмов;
* названия студий и компаний;
* логотипы;
* фирменные эмблемы;
* узнаваемые костюмы;
* точная композиция постеров и кадров.

Цель fallback — не сделать нейтральную картинку «про кино вообще», а сохранить максимум визуального смысла конкретной новости без восстановления удалённой идентичности.

Примеры универсальных semantic-признаков:

```text
winter fantasy
→ снег
→ лёд
→ сказочный пейзаж
→ северное сияние
→ холодная цветовая палитра

marine adventure
→ море
→ побережье
→ парусное судно
→ горизонт

science fiction
→ футуристическая архитектура
→ исследовательская среда
→ световая энергия

corporate/regulatory news
→ студийный офис
→ документы
→ переговоры
→ киноиндустриальный контекст
```

---

## Image retry budget

Retry budget считается отдельно для конкретного:

```text
batch
generated_post
request_kind
prompt_version
```

Поэтому новая replacement combination получает:

* новый batch;
* новый generated_post;
* новый image request key;
* новый независимый image retry budget.

Текущая production-логика допускает:

```text
1 normal image attempt
+
до 2 semantic fallback attempts
```

для одной TOP-3 combination.

---

# TOP-3 replacement cascade

Если semantic fallback также устойчиво блокируется модерацией, daily workflow не обязан завершаться ошибкой.

Вместо этого система выбирает следующую сохранённую ranking combination.

Текущий production limit:

```text
winner
+
до 3 replacement combinations
```

То есть в одном ежедневном workflow может быть использовано до четырёх разных TOP-3 combinations.

Для каждой новой combination выполняются заново:

```text
text generation
→ self-review
→ image cascade
```

Повторный ranking при этом не требуется.

---

## Atomic replacement transition

Replacement выполняется одной PostgreSQL-транзакцией.

Старые артефакты:

```text
generated_post → superseded
publication_batch → superseded
selection_attempt → moderation_blocked
```

Создаётся:

```text
new selection_attempt → active
```

Daily workflow возвращается в:

```text
workflow_status = running
current_stage   = generation

batch_id            = NULL
generated_post_id   = NULL
image_generation_id = NULL
```

`ranking_run_id` сохраняется.

После этого orchestrator продолжает работу с новой combination.

---

## Restart-safe replacement history

Каждый использованный TOP-3 сохраняется в:

```text
daily_workflow_selection_attempts
```

История содержит:

* workflow;
* ranking run;
* combination;
* attempt number;
* winner/replacement;
* source selection;
* active/moderation_blocked/ready_for_review;
* batch;
* generated post;
* image generation;
* время завершения.

Это необходимо, чтобы после рестарта процесс продолжался с уже выбранной replacement combination и не возвращался к winner.

---

# Гарантия ежедневной публикации и hard fallback

Целевое требование проекта:

> **Ежедневная публикация не должна отсутствовать только потому, что Image API отказал в генерации изображения по moderation.**

Текущий многоступенчатый cascade уже существенно снижает вероятность такого отказа:

```text
normal image
→ semantic fallback
→ replacement TOP-3
→ новый semantic fallback
→ следующие replacement combinations
```

Однако абсолютную техническую гарантию нельзя честно связывать только с внешним Image API: любой внешний API может быть недоступен, вернуть ошибку инфраструктуры или отклонить все запросы.

Поэтому для **буквальной гарантии отсутствия moderation dead-end** целевая архитектура предусматривает последний fallback, не зависящий от генеративного Image API:

```text
Image API cascade exhausted
        ↓
deterministic local PNG hard fallback
        ↓
готовое native Telegram image
        ↓
publication
```

Hard fallback должен строиться локально из уже известных безопасных данных, например как оригинальная типографическая/атмосферная editorial card:

```text
название или безопасный заголовок новости
+
жанровая палитра
+
простые абстрактные/геометрические мотивы
+
три зоны
```

Он не требует локальной LLM или ML-модели.

После реализации этого последнего слоя можно будет гарантировать, что **именно отказ image moderation не остановит публикацию**.

Эта гарантия не означает защиту от внешних инфраструктурных аварий вроде полной недоступности Telegram, PostgreSQL, сервера или сети.

---

# Daily workflow и checkpoints

Daily production имеет собственную persisted state machine.

Основные стадии:

```text
reserved
ranking
generation
image
review_delivery
awaiting_review
failed
```

Ключевые identity/checkpoints:

```text
ranking_run_id
batch_id
generated_post_id
image_generation_id
```

Все checkpoints сохраняются в PostgreSQL.

Это позволяет безопасно восстанавливать workflow после рестарта процесса или сервера без повторных платных вызовов, если их outcome уже известен.

---

# Idempotency и uncertain outcomes

Для внешних операций используются request keys и persisted reservations.

Общий принцип:

```text
reserve in DB
        ↓
external API call
        ↓
persist result
```

Если после reservation процесс видит состояние, для которого невозможно доказать результат внешнего запроса, он не делает слепой повтор.

Такие состояния считаются orphan/uncertain и требуют отдельной recovery-логики.

Этот подход используется для:

* ranking;
* text generation;
* image generation;
* Telegram review delivery;
* final publication.

---

# PostgreSQL как authoritative state

База данных является источником истины для production workflow.

В PostgreSQL сохраняются:

* RSS sources;
* news items;
* collection runs;
* ranking runs;
* ranking events;
* news scores;
* ranking combinations;
* generation batches;
* batch items;
* generated posts;
* image generation requests;
* daily workflow runs;
* daily workflow selection attempts;
* review actions;
* publication attempts;
* Telegram publication metadata;
* версии prompt/model/formula;
* error/recovery telemetry.

Это позволяет:

* воспроизводить решения;
* разбирать сбои;
* не терять состояние после рестарта;
* блокировать дубли;
* отслеживать стоимость;
* проводить production diagnostics.

---

# Статусы

## Publication batch

Основные статусы:

```text
draft
ranked
generated
awaiting_review
approved
rejected
superseded
publishing
published
failed
```

`superseded` означает, что batch был корректно заменён новым TOP-3 внутри того же daily workflow.

---

## Generated post

Generated post также может быть переведён в:

```text
superseded
```

если соответствующий TOP-3 заменён после устойчивого image moderation failure.

---

## Image generation request

Основные состояния:

```text
reserved
completed
failed
```

Для failed request сохраняются:

* тип ошибки;
* сообщение;
* время ошибки;
* request kind;
* prompt version;
* model;
* request payload;
* telemetry.

---

# Ручной review — текущий переходный режим

На текущем этапе после успешной подготовки текста и PNG пост отправляется ответственному пользователю как:

```text
native Telegram photo
+
caption
+
inline buttons
```

Доступные действия:

```text
✅ Approve
✏️ Change
❌ Reject
```

Результаты review сохраняются в PostgreSQL.

Approved post публикуется отдельным publisher через native `sendPhoto + caption`.

Это переходный слой.

**Целевая архитектура проекта — автоматическая публикация без обязательного человеческого подтверждения.**

Перед отключением human gate должны быть накоплены production-метрики по:

* точности ranking;
* качеству self-review;
* factual correctness;
* caption limits;
* image moderation recovery;
* image semantic quality;
* duplicate prevention;
* publication reliability;
* orphan recovery.

---

# Final publication

Финальная публикация выполняется через Telegram Bot API как native photo message.

Сохраняются:

* generated post;
* publication attempt;
* Telegram chat ID;
* Telegram message ID;
* request payload;
* response payload;
* publication status;
* error data.

Если Telegram отправил сообщение, но процесс не смог надёжно завершить DB-finalization, состояние должно считаться `unknown`, а не автоматически повторяться.

Это защищает канал от случайных дублей.

---

# Структура проекта

```text
top3-news-bot/
├── app/
│   ├── bot/
│   │   ├── handlers.py
│   │   ├── review_delivery_service.py
│   │   └── ...
│   │
│   ├── collectors/
│   │   └── ...                     # RSS collectors
│   │
│   ├── db/
│   │   ├── daily_workflow.py
│   │   ├── daily_workflow_checkpoints.py
│   │   ├── daily_workflow_selection_attempts.py
│   │   ├── daily_workflow_replacement.py
│   │   ├── generation_selection.py
│   │   ├── generation_reservation.py
│   │   ├── image_generation_reservation.py
│   │   ├── approved_publications.py
│   │   └── ...
│   │
│   ├── generation/
│   │   ├── openai_generator.py
│   │   ├── openai_pipeline.py
│   │   ├── image_generator.py
│   │   ├── openai_image_pipeline.py
│   │   ├── openai_image_factory.py
│   │   ├── official_trailer_enrichment.py
│   │   └── ...
│   │
│   ├── publication/
│   │   ├── approved_service.py
│   │   └── ...
│   │
│   ├── ranking/
│   │   ├── openai_event_pipeline.py
│   │   ├── full_formula.py
│   │   └── ...
│   │
│   ├── review/
│   │   └── ...
│   │
│   └── workflows/
│       └── daily_production.py      # основной daily orchestrator
│
├── config/
│   └── systemd/
│       ├── top3-news-bot.service
│       ├── top3-news-collector.service
│       ├── top3-news-collector.timer
│       ├── top3-news-cleanup.service
│       ├── top3-news-cleanup.timer
│       ├── top3-news-daily.service
│       └── top3-news-daily.timer
│
├── data/
│   └── images/
│       ├── generated/
│       └── reference/
│
├── logs/
├── migrations/
├── prompts/
├── scripts/
├── tests/
├── .env.example
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

Структура развивается по мере появления persisted workflow, recovery и idempotency-механизмов.

---

# Основные production-компоненты

## Collectors

Отвечают за:

* RSS;
* расписание сбора;
* получение исходных материалов;
* сохранение raw news items.

---

## Ranking

Отвечает за:

* 24-часовое окно;
* eligibility;
* OpenAI scoring;
* формулу;
* diversity;
* ranking combinations;
* winner TOP-3.

---

## Generation

Отвечает за:

* Telegram-текст;
* self-review;
* official trailer enrichment;
* image prompt;
* semantic fallback;
* OpenAI Images API.

---

## Workflows

`app/workflows/daily_production.py` является production orchestrator.

Он координирует:

```text
ranking
→ selection
→ generation
→ image
→ replacement
→ review/publication
```

Orchestrator не должен хранить authoritative state только в памяти.

Все критические checkpoints записываются в PostgreSQL.

---

## Publication

Отвечает за:

* native Telegram photo + caption;
* idempotent publication attempts;
* Telegram response;
* final publication state;
* защиту от дублей.

---

# Среды проекта

## Разработка

Основная разработка:

```text
/home/mihail/d/films/movie_news/top3-news-bot
```

Среда:

```text
Linux
Cursor
Python 3.12
uv
Git
```

На ноутбуке выполняются:

* редактирование;
* syntax checks;
* статические проверки;
* Git.

---

## Production

Сервер:

```text
Ubuntu 24.04.3 LTS
```

Каталог:

```text
/opt/top3-news-bot
```

Production tests и реальные pipeline runs выполняются на сервере.

---

# Deployment

Схема:

```text
Cursor
   ↓
local Git
   ↓
GitHub main
   ↓
scripts/deploy_server.sh
   ↓
cloud-001
   ↓
systemd services/timers
```

Deploy script:

* проверяет clean server working tree;
* получает `origin/main`;
* выполняет fast-forward;
* синхронизирует Python dependencies;
* проверяет Python syntax;
* обновляет systemd units при необходимости;
* рестартует bot service;
* включает и рестартует timers;
* выводит deployed commit и состояние сервисов.

---

# Технологический стек

```text
Ubuntu Linux
Python 3.12
uv
asyncio
aiogram
asyncpg
PostgreSQL 16
OpenAI API
OpenAI Responses API
OpenAI Images API
Telegram Bot API
systemd
Bash
Git
GitHub
```

---

# Конфигурация и секреты

Локальные и production-параметры:

```text
.env
```

`.env` не хранится в Git.

В repository находится только:

```text
.env.example
```

Запрещено сохранять в repository:

* OpenAI API keys;
* Telegram bot token;
* PostgreSQL passwords;
* private SSH keys;
* другие production secrets.

---

# Версионирование AI-компонентов

Для воспроизводимости фиксируются:

```text
model name
prompt version
generator version
formula version
request key version
workflow version
```

Изменение значимого AI-поведения должно сопровождаться новой версией prompt/generator, если старые request keys и retry budget не должны смешиваться с новой логикой.

Это особенно важно для image fallback:

```text
normal prompt version
!=
semantic fallback prompt version
```

---

# Тестирование

В проекте используются отдельные diagnostic/integration scripts.

Критические production-механизмы проверяются без реальных платных запросов там, где это возможно:

* idempotency;
* reservation;
* restart recovery;
* ranking combinations;
* explicit combination text generation;
* atomic replacement;
* moderation retry budget;
* transition к следующему TOP-3;
* Telegram publication state.

Для image quality дополнительно используются изолированные live tests с реальным Image API без записи в production workflow и без Telegram publication.

---

# Текущий уровень готовности

Уже реализовано:

```text
RSS collection
PostgreSQL persistence
OpenAI ranking
ranking combinations
TOP-3 selection
text generation
automatic self-review
official trailer enrichment
semantic image generation
moderation-safe fallback
image retry budget
TOP-3 replacement cascade
restart-safe selection history
daily workflow checkpoints
native Telegram review
approved native-photo publisher
daily systemd timer
idempotency and duplicate protection
```

Переходный этап:

```text
automatic production
        ↓
human review
        ↓
publication
```

Целевая версия:

```text
automatic production
        ↓
automatic quality gates
        ↓
hard image fallback if needed
        ↓
automatic publication
```

---

# Ближайшие задачи

## 1. Наблюдение за production cascade

Накопить реальные результаты:

* normal image success rate;
* semantic fallback success rate;
* количество fallback attempts;
* частоту TOP-3 replacements;
* качество semantic fallback;
* moderation failure rate.

---

## 2. Hard image fallback

Реализовать последний deterministic fallback, не зависящий от Image API.

Цель:

```text
никакой moderation_blocked
не может оставить daily post
без native Telegram image
```

---

## 3. Automatic publication gate

После накопления статистики определить условия, при которых daily workflow автоматически переходит:

```text
ready
→ publishing
→ published
```

без human approval.

---

## 4. Monitoring и alerting

Добавить production monitoring:

* missed daily run;
* workflow failed;
* orphan state;
* Telegram unavailable;
* PostgreSQL error;
* unusual OpenAI failure;
* excessive replacement count;
* publication not confirmed.

---

## 5. Full-auto режим

Финальная цель:

```text
RSS
→ ranking
→ TOP-3
→ text
→ self-review
→ image cascade
→ automatic quality gate
→ Telegram publication
```

без участия человека в штатной работе.

Человек должен оставаться оператором системы, а не обязательным этапом ежедневной публикации.