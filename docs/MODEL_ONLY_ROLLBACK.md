# Откат Image Prompt Agent и обновление моделей — 2026-10-06

По результатам production изображений пользователь попросил вернуть реализацию
к **3309878393254fb4ef4d9b02271fb02b9c9b0633**, затем обновить только модели.
Коммит отката **9474f69** имеет ровно то же Git tree, что 3309878; история main
сохранена. Финальное приложение отличается от 3309878 только тремя файлами:

- `app/config.py`: ranking и generation по умолчанию **gpt-6.1-sol**, image —
  **gpt-image-2.5-flare**;
- `app/ranking/openai_usage.py`: добавлен тариф gpt-6.1-sol;
- `app/generation/image_openai_usage.py`: добавлен тариф gpt-image-2.5-flare.

Прежние model pricing entries сохранены для исторических запросов. Тарифы
сверены 2026-10-06 с официальными страницами:
[GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol),
[GPT-Image-2.5 Flare](https://developers.openai.com/api/docs/models/gpt-image-2.5-flare).
Responses API и Images API, существующие параметры запросов и reasoning,
prompt версии, selection/ranking pipeline, встроенные image instructions и
fallback/retry поведение возвращены к 3309878. Image quality — прежний medium,
размер — 1024×1024. Отдельного платного Image Prompt Agent больше нет.

Применённая migration 019, nullable image_prompt_id и исторические строки
image_prompt_plans остаются в production БД. Старый код не читает и не пишет
планы; INSERT image requests оставляет nullable ссылку пустой. SQL 019 сохранён
как история схемы. Ручной файл image-prompt-moderation-recovery-examples.txt
сохранён под согласованным именем и больше не загружается приложением.

Разработка — cloud-002, затем GitHub; все тесты — cloud-001. Проверки используют
отдельный временный checkout и изолированную PostgreSQL на 55432 с read-only
snapshot production. Платный smoke и ручной production выпуск не требуются:
следующие scheduled выпуски пользователь оценивает в Telegram.

`scripts/rollout_model_only.sh` получает целевой commit из GitHub, проверяет
clean tree, idle daily service, uv/dependencies и прежний PNG asset, делает
приватный backup .env, обновляет три model settings, убирает настройки агента
и запускает штатный deploy из целевого commit. Внешние .env/API/Telegram
секреты не выводятся. Migration rollback и удаление production данных отсутствуют.
До запуска rollout production оставался на 05ad8f5.

## Проверка на cloud-001

Candidate **4b7c981** проверен в `/tmp/top3-model-only-tests-20261006/work`:
PostgreSQL 16 на порту 55432, отдельная database top3_news_test; snapshot
production экспортирован read-only. Dummy credentials и socket guard разрешают
тестам только подключения к изолированной БД. Live/API/publication probes
исключены из suite. Production данные не изменялись.

Offline suite: **69 scripts**, **57 PASS**, **12 FAIL**. Все 12 падений повторены
на отдельном checkout исходного **3309878** в той же тестовой среде:

| Тест | Ограничение baseline/fixture |
| --- | --- |
| test_daily_workflow_checkpoints | исторический batch использует другой channel ID, чем dummy test settings |
| test_daily_workflow_lifecycle | исторический workflow уже существует с другим channel ID |
| test_generation_reservation | отсутствует исторический ranking_run_id=18 |
| test_generation_revision_reservation | отсутствует исторический ranking_run_id=18 |
| test_generation_revision_request_key | test changed version совпадает с текущей v5 |
| test_local_ranking_pipeline | старое окно 2026-07-31 не содержит кандидатов |
| test_openai_pipeline | старое окно 2026-07-31 не содержит кандидатов |
| test_openai_event_diagnostic_failure | старое окно не достигает ожидаемой ветки |
| test_openai_image_pipeline | baseline fixture 64×96 не проходит уже существующий square guard |
| test_ranking_run_reservation | отсутствуют исторические news_id=7,8 |
| test_review_delivery_lifecycle | старый fixture post уже имеет review delivery |
| test_official_trailer_enrichment | старое ожидаемое reason для отсутствующего iframe |

Общая suite не полностью зелёная; новых падений относительно baseline не найдено.
Пройдены image factory/reservation/completion/storage и daily moderation fallback,
text/event factories и clients, generation/revisions, cost/usage, trailer contracts,
publication/review components с fake Telegram. Пройдены compileall и bash syntax.

Отдельная проверка runtime под запретом сети подтвердила:

- обычный ranking, event ranking и text generation выбирают gpt-6.1-sol;
- image factory отправляет gpt-image-2.5-flare/medium/1024×1024 и **точный прежний
  встроенный prompt**, без prompt_agent; fake SDK получил ровно один Images call;
- новый text tariff считает cache/input/output cost, новый image tariff доступен;
- новый prompt на минимальном TOP-3 имеет 15205 символов и укладывается в
  [лимит Images API 32000 символов](https://developers.openai.com/api/reference/resources/images/methods/generate).

Платных API вызовов и Telegram отправок нет. Логи и JSON summaries лежат в
`test-logs/` и `baseline-logs/` на cloud-001; после проверки тестовая PostgreSQL
остановлена, snapshot и её данные удалены. Применённая production migration 019
оставлена как история; rollback script не выполняет SQL миграций.

## Production подтверждён

2026-10-06 rollout **ac87f8af9b91b47063231920076995409667ab99** завершён.
Независимая SSH/read-only проверка подтвердила clean tree и точное совпадение
image generator/factory/pipeline и daily workflow с 3309878. Отличия приложения
от baseline ограничены указанными выше config и двумя pricing registries.

Runtime использует gpt-6.1-sol для ranking/text и gpt-image-2.5-flare для
изображений, medium, 1024×1024; отдельного агента нет, прежний static fallback
PNG доступен. Migration 019 и две исторические prompt plan записи сохранены.
Приватная backup .env: `/tmp/top3-model-only-rollout.dKMW6c`.

Bot перезапущен **2026-10-06 11:28:00 UTC**; PostgreSQL pool и Telegram polling
стартовали успешно, Result=success, NRestarts=0. Bot и collector/cleanup/daily
timers active. Daily oneshot inactive между выпусками; следующий автоматический
выпуск — **2026-10-07 07:30 UTC**. Проверка не вызывала платные API и не запускала
ручной выпуск. Первый scheduled выпуск после отката ещё предстоит.
