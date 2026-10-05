# Image Prompt Agent v1

## Граница изменения

Новая стадия визуального планирования работает перед каждым первоначальным
Image API request и перед редакционной перегенерацией. Ranking, выбор TOP-3,
генерация текста, его compaction и проверка официальных trailer URLs сохраняют
существующую реализацию и модели. Их миграция на другую модель — отдельная задача.

Аудит 2026-10-04 подтвердил исходное состояние: commit `3309878`, Responses API
в текстовых компонентах, `gpt-6-sol` для ranking/generation, `gpt-image-2` для
изображений. PostgreSQL production имела миграции 001–018.

## Карта текущего и нового потока

Ранее `run_reserved_openai_image_generation` строил длинный детерминированный
image prompt прямо из selection, вычислял image_request_key, резервировал строку
image_generation_requests, вызывал Image API и сохранял PNG/SHA256. После
moderation_blocked daily workflow переключал generator на старый fallback prompt.

Теперь защищённый pipeline выполняет:

1. Проверяет batch/post, текущий TOP-3 и редакционный context существующими
   валидаторами image reservation; читает финальный `generated_posts.post_text`,
   generated_items и оригинальные тексты выбранных новостей.
2. Резервирует `image_prompt_plans` до платного Responses API request.
3. Получает strict `ImagePromptPlan` от `gpt-6.1-sol`, Responses API, `store=False`,
   reasoning low, максимум 5000 output tokens, без web search и SDK retries.
4. Сохраняет exact final_image_prompt, три block plans, mode, версии, parent,
   usage, cost и response/request IDs. Транзакция завершена до Image API.
5. Строит прежний image_request_key и резервирует image_generation_requests
   с `image_prompt_id`; FK и exact prompt проверяются в этой транзакции.
6. Отправляет в Image API только final_image_prompt:
   `gpt-image-2.5-flare`, medium, 1024x1024, PNG, opaque, moderation auto, n=1.
7. Сохраняет PNG, SHA256, image usage/cost и прежние generated_posts image fields.
8. При доказанном moderation block создаёт новый recovery plan с parent связями,
   предыдущим exact prompt/structured plan, диагностикой и manual examples.

Основные модули:

- `app/generation/image_prompt_agent.py`: Responses client и строгий контракт;
- `app/db/image_prompt_plans.py`: reservation, idempotency, persistence;
- `app/generation/moderation_diagnostics.py`: фактические optional поля ошибки;
- `app/generation/openai_image_pipeline.py`: точка интеграции;
- `app/generation/editorial_image_fallback.py`: бесплатная финальная карточка.

## Режимы и retry budget

NORMAL_CREATIVE изолирован от recovery examples. Он сохраняет выразительную
киноиндустриальную концепцию и художественную свободу image model. Recovery
получает реальные примеры из `image-prompt-moderation-recovery-examples.txt`.
Сохранены все исходные примеры; добавлено только пояснение об их назначении.

Правка инструкций 2026-10-05 по первому production изображению: естественная
цветопередача и реалистичная композиция вместо акцента на яркости/насыщенности;
без выдуманных элементов ради контраста и без мультяшности, если новость не
про мультфильм. Блоки не нумеруются на изображении; светлые разделители —
примерно 4 px при 1024×1024. Агент должен перенести эти требования в
final_image_prompt. Recovery соблюдает те же правила, а прежний prompt и
исторические примеры не возвращают чрезмерную насыщенность или нумерацию.
Файлы инструкций читаются при каждом build_input, их точный текст сохраняется
в plan и участвует в request key; обновление не требует restart сервиса.

MODERATION_RECOVERY меняет способ визуализации. Проверка отвергает идентичный
prompt и повтор всех трёх strategy; она не доказывает художественное качество
или прохождение модерации. Финальный prompt ограничен 2200 символами и 4000
байтами UTF-8. В image request не попадают инструкции агента, исходные тексты,
примеры или история отказов.

Фактическая исходная state machine разрешала NORMAL и **до двух** попыток
fallback (`MAX_IMAGE_ATTEMPTS_PER_PROMPT_VERSION=2`). Этот лимит сохранён:
NORMAL → RECOVERY → RECOVERY. Полноценный SAFE_EDITORIAL_FALLBACK предусмотрен
в контракте, но автоматически не запускается; дополнительной платной попытки нет.

При исчерпании существующего бюджета daily workflow создаёт локальную PNG-карточку
с тремя конкретными заголовками выбранных новостей. Она заменяет прежний
универсальный PNG, сохраняет исходные TOP-3/batch/post и не вызывает OpenAI.
Установка Pillow и системные DejaVu fonts используются из server environment.

## Состояния PostgreSQL и идемпотентность

Миграция 019 добавляет одну таблицу image_prompt_plans и nullable FK
image_generation_requests.image_prompt_id. Исторические image requests сохраняются.
Request kinds image state machine остаются `initial` и `regenerate`; отдельное
поле plan.attempt_kind различает initial, moderation_recovery, editorial_revision
и зарезервированный safe_editorial_fallback.

Prompt: reserved → completed / failed. Резервирование использует тот же advisory
lock generated_post_id, что и image reservation. Active key и одна reserved
запись на post защищены уникальными индексами. Конкурентный/незавершённый
reserved агент не повторяется автоматически. Completed план без image reservation
можно использовать повторно после crash; completed/reserved image не вызывает
ни агент, ни image model второй раз. Image failure не удаляет completed prompt.
Recovery после нового moderation result получает новый key и source_prompt_id /
source_image_generation_id. Не-moderation image retry переиспользует прежний plan.

При сбое агента image reservation ещё не существует. Ошибка сохраняется в prompt
history; если API usage получена, стоимость сохраняется даже при invalid output.
Неоднозначный transport failure не считается разрешением на moderation recovery.

Exact prompt проверяется повторно при image reservation; изменение финального
post_text между двумя стадиями блокирует Image API. Редакционная перегенерация
создаёт отдельный plan с attempt_kind=editorial_revision и source_prompt_id.
Обычная text revision сохраняет существующую семантику наследования картинки.

## Диагностика

`image_generation_requests.response_metadata.openai_error` хранит error_code,
request_id, moderation_stage и moderation_categories. Поля отсутствующей
диагностики имеют JSON null. Stage интерпретируется как input/output только
если это явно вернул API; явное unknown также сохраняется, а отсутствующее поле
остаётся null. Причина блока не выводится
из содержания новости. Отклонённый output PNG сохранить невозможно, когда
Image API не возвращает его приложению.

```sql
SELECT p.image_prompt_id, p.prompt_mode, p.attempt_kind, p.attempt_number,
       p.source_prompt_id, p.agent_model, p.final_image_prompt, p.openai_cost,
       r.image_generation_id, r.model_name, r.image_status,
       r.response_metadata -> 'openai_error' AS diagnostics
FROM top3_news.image_prompt_plans p
LEFT JOIN top3_news.image_generation_requests r USING (image_prompt_id)
WHERE p.batch_id = :batch_id
ORDER BY p.image_prompt_id, r.image_generation_id;
```

## Тарифы

Добавлены новые entries с pricing_version=2026-10-04; прежние тарифы не меняются.
Standard USD / 1M tokens:

| Модель | Text input | Cached text input | Output |
| --- | ---: | ---: | ---: |
| gpt-6.1-sol | 2.00 | 0.10 | 10.00 text |
| gpt-image-2.5-flare | 5.00 | 1.25 | 30.00 image |

Image input: 8.00, cached image input: 2.00. Image cost учитывает фактически
возвращённые modality tokens по стандартным ставкам, как прежний registry.
Cached image input SDK не выделяет в существующем usage contract, поэтому
image estimate остаётся conservative uncached estimate.

Источники, проверены 2026-10-04:

- https://developers.openai.com/api/docs/models/gpt-6.1-sol
- https://developers.openai.com/api/docs/models/gpt-image-2.5-flare
- https://developers.openai.com/api/docs/guides/image-generation

## Проверки и rollout

Все тесты выполняются на cloud-001. Исходники разрабатываются на nl-cloud-002;
до deploy код тестируется в отдельном временном каталоге cloud-001, на изолированной
PostgreSQL порта 55432, с read-only копией production schema/data. Тестовая среда
не содержит действующих OpenAI/Telegram credentials. Offline suite запрещает
внешние socket connections. Скрипты реальной публикации и live API probes
не входят в бесплатную suite.

Новый тест: `python -m scripts.test_image_prompt_agent`. Проверяет strict schema,
NORMAL/recovery isolation, сохранение до Image API, parent chains, nullable
diagnostics, concurrent reservation, completed reuse, invalid output telemetry
и editorial revision. Существующие image reservation/storage/completion,
retry budget, text revisions, publication и cleanup проходят отдельные regression
scripts на той же изолированной серверной БД.

Rollout: Git/GitHub → миграция 019 от владельца michael_psql → штатный
scripts/deploy_server.sh → environment-specific OPENAI_IMAGE_MODEL,
OPENAI_IMAGE_PROMPT_AGENT_MODEL, OPENAI_IMAGE_QUALITY → проверка сервиса/таймеров
и контролируемый live image smoke в пределах разрешённого $1 на выпуск.
Рабочий daily workflow не запускается вручную ради smoke и тестовые сообщения
не отправляются в Telegram. Следующий production выпуск оценивает пользователь.
