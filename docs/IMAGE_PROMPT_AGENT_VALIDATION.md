# Проверка Image Prompt Agent v1 — 2026-10-04

Все исполняемые проверки проведены на cloud-001. Production БД не изменялась:
offline tests использовали её read-only snapshot в отдельной PostgreSQL 16,
порт 55432, database top3_news_test, отдельный временный checkout.

## Бесплатные проверки

Полная доступная suite без live API/publication probes: **70 scripts**, из них
**60 PASS**, **10 baseline failures**. Offline socket guard разрешал только
подключение к изолированной PostgreSQL. Production fixtures в тестовой копии
после mutation-сценариев откатывались транзакциями; новые fixtures удалялись.

Пройдены новые agent tests и связанные regression:

- ImagePromptPlan strict schema, три news_id/позиции, NORMAL/recovery isolation;
- exact prompt saved before Image API, request FK, parent цепочка recovery;
- concurrent reservation, orphan blocking, completed reuse;
- reservation rollback и запрет подмены сохранённого prompt;
- failure history и usage/cost при invalid structured output;
- editorial revision отдельно от moderation retry;
- image factory, reservation, completion, PNG storage/SHA256;
- daily moderation retry budget и финальная локальная карточка;
- text generation/revisions, integrity repair/compaction, trailer contracts;
- fake Telegram native photo publication, review lifecycle service и cleanup;
- syntax compileall и bash syntax rollout script.

10 старых failures воспроизведены также на исходном commit **3309878**,
до изменений Image Prompt Agent:

| Тест | Причина |
| --- | --- |
| test_daily_workflow_lifecycle | фиксированная дата уже занята существующим workflow |
| test_generation_reservation | исторический ranking_run_id=18 отсутствует |
| test_generation_revision_reservation | исторический ranking_run_id=18 отсутствует |
| test_generation_revision_request_key | changed prompt version совпадает с текущей v5 |
| test_local_ranking_pipeline | историческое окно 2026-07-31 не содержит кандидатов |
| test_openai_pipeline | историческое окно 2026-07-31 не содержит кандидатов |
| test_ranking_run_reservation | удалены исторические news_id=7,8 |
| test_openai_event_diagnostic_failure | историческое окно не достигает ожидаемой insufficient-TOP3 ветки |
| test_review_delivery_lifecycle | выбранный существующий post уже имеет review delivery |
| test_official_trailer_enrichment | устаревшее ожидаемое reason для отсутствующего iframe |

Эти тесты и production-код их компонентов не менялись в рамках image stage.
Полностью зелёной общей suite этот результат не является. Изменения в двух
существующих image tests актуализировали square PNG fixture и current prompt
version assertion; новый factory test передаёт готовый короткий prompt.

## Платный smoke

Один реальный сохранённый TOP-3: source batch **143**, news_id **6112, 6115, 6129**.
Тестовый batch **190**, prompt **24**, image request **184** существовали только
в изолированной БД и после экспорта trace были удалены.

| Стадия | Модель | Input tokens | Output tokens | Estimated USD |
| --- | --- | ---: | ---: | ---: |
| Prompt Agent | gpt-6.1-sol | 2576 | 948 | 0.01591850 |
| Image API | gpt-image-2.5-flare | 533 | 439 image | 0.01583500 |
| Всего | | | | **0.03175350** |

Agent input включает 2573 cache-write tokens; они оценены по официальной ставке
1.25 × standard input. Это estimated model cost по usage, не отдельная выписка
из billing dashboard. SDK retries=0, один agent request и один image request,
Telegram calls=0. Recovery проверялся бесплатно, через synthetic moderation error.

Оба Responses/Image requests завершились completed. Получен PNG 1024×1024,
medium, opaque, moderation auto. Exact prompt и structured plan сохранены раньше
image reservation. Image API получил только final_image_prompt.

SHA256 PNG:

`31ef854548e7b856950cacc8a66db31f01ce72ad55cee84e44183e598d314b4f`

На cloud-001: `/tmp/top3-image-agent-tests-20261004/live-smoke/`.
Копия результатов на dev: `data/images/image-prompt-agent-smoke/` (ignored by Git).
Trace содержит полный prompt, plan, original news input, SDK request IDs,
usage/cost и image request payload; секретов в trace нет.

## Production rollout

На момент подготовки rollout production оставался на `3309878`, миграциях
001–018 и gpt-image-2. SSH michael доступен; passwordless sudo и настроенное
подключение владельца michael_psql отсутствуют. Поэтому применение migration 019
и штатный restart требуют интерактивного sudo на cloud-001.

Подготовлен `scripts/rollout_image_prompt_agent.sh`: проверяет exact origin/main
commit, clean tracked working tree и отсутствие работающего daily workflow,
делает приватную backup .env/schema, применяет migration от владельца схемы,
меняет только image-related .env settings, запускает штатный deploy script и
проверяет commit/migration/timer. Production выполнение ещё не подтверждено.

Первая попытка rollout 2026-10-04 остановилась до применения SQL: приватная
backup directory (0700, владелец michael) не позволяла psql от postgres читать
019.sql через `-f path`. Production остался на 3309878, миграциях 001–018 и
gpt-image-2; bot и daily timer active. Исправлено: shell текущего пользователя
открывает файл и передаёт SQL через stdin (`-f - < file`), сохраняя приватные
права backup и SET ROLE в той же psql session. Повторный rollout ещё не подтверждён.
На cloud-001 пройдены bash syntax check и read-only проверка реальным psql:
SET ROLE и SQL из stdin выполняются в одной сессии; transaction ROLLBACK.
Исправление готовится на cloud-002, публикуется в GitHub, rollout на cloud-001
извлекается из fetched origin/main, без прямого копирования production-кода.
