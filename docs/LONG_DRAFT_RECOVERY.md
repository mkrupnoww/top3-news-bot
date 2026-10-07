# Восстановление выпуска 2026-10-07

Таймер стартовал в 07:30 UTC. Ranking 235 завершился; workflow 82 / batch 146
упал в 07:39 на self-review: модель gpt-6.1-sol вернула post_text длиннее 1000
символов. Strict Pydantic parsing остановил процесс раньше канонической сборки
и существующего local compactor; изображение и Telegram review не вызывались.

Ранее похожее переполнение уже обрабатывалось в post_integrity.py:
`_compact_items_to_post_limit` и `build_deterministic_integrity_fallback`.
Регрессия 2026-09-22 сохранена в test_text_integrity_repair._overflow_payload.

Исправление сохраняет публичный контракт 1000 символов. Только когда все
validation errors относятся к длине post_text, структура повторно валидируется,
текст собирается из items и при переполнении применяется прежний compactor.
Порядок/IDs, trailer metadata, headline/body limits и ошибки структуры не
ослабляются. Финальный payload снова проверяется strict model. Модельный ответ
остаётся в telemetry; исходная строка не обрезается для публикации.
Generator version v9 создаёт новый request key и оставляет failed batch в истории.

Manual retry: `python -m scripts.retry_failed_text_workflow --workflow-id 82
--ledger-directory /tmp/top3-text-retry-20261007 --max-usd 1`.
Явно возобновляется только text-schema failure без generated_post/image.
Saved ranking, selection, publication_date и as_of сохраняются; ranking API
запрещён guard. Предыдущее состояние экспортируется до UPDATE, старый batch
не удаляется. CLI завершает обычный workflow до native Telegram review.

Каждый платный запрос сначала резервирует conservative allowance в отдельном
cost ledger. SDK retries=0; Responses ограничены 5000 output tokens, low reasoning
и одним tool call. Неизвестный расход после transport/API error сохраняет
allowance; следующий запрос запрещается, если allowance не помещается в $1.
После completed API расход учитывается по usage и tariff, включая web-search fee.
Это оценка API usage, а не отдельная billing invoice.
Источники: [Responses limits](https://developers.openai.com/api/reference/cli/resources/responses/methods/create),
[Image pricing](https://developers.openai.com/api/docs/models/gpt-image-2.5-flare).

Проверки выполнены только на cloud-001, checkout из GitHub, PostgreSQL 55432
с read-only production snapshot, dummy credentials и запретом внешней сети.
Пройдены regression long draft/overflow, existing integrity repair/post generator/
generation client/factory/request-key/completion/revision completion и manual
retry history/budget. Production запуск ещё не выполнен на этапе подготовки.

## Результат ручного запуска

2026-10-07 manual retry выполнен на cloud-001 без запуска таймера. Ranking 235,
combination 54034, news IDs 6319/6387/6320 и исходный cutoff сохранены.
Новый batch 147 / post 148 успешно прошёл text generation/self-review; итог —
890 символов. Исходный failed batch 146 остаётся в истории.

Первый Image API запрос отклонён output moderation (категория `other`,
request ID `req_5a14e8fc29f24617a032d2d799f78886`). Ответ не содержит usage.
Image 147 сохранён как failed. Перед следующим Image API вызовом бюджетный
guard остановил retry: 0.652107 USD уже учтено/зарезервировано, следующий
allowance 0.499025 USD превышал оставшийся бюджет. Image 148 failed;
для этой записи API не вызывался.

Отдельный `complete_budget_retry_with_local_image.py` завершил image stage
существующим локальным PNG fallback, без OpenAI client и платных запросов.
Скрипт проверяет identity/state, наличие PNG и отсутствие delivery attempts,
сохраняет аудит до UPDATE, использует workflow lock и штатные image completion /
Telegram delivery routines. Его read-only preflight и syntax check выполнены
на cloud-001 до запуска.

Image 149 completed (`local_static_png_asset`, `movie_news_image_versatile_option_v2`).
Telegram review delivery 63 имеет status sent / message 446; workflow 82 и
post 148 — awaiting_review. Это отправка редактору на проверку; публикация
в канале остаётся штатным следующим действием редактора.

Ledger `/tmp/top3-text-retry-20261007/cost-ledger.json`: два Responses вызова,
usage estimate 0.012886 + 0.046896 = **0.059782 USD**; один Image API запрос
без usage, его conservative allowance **0.592325 USD** сохранён. Всего
учтено/зарезервировано **0.652107 USD**, лимит 1 USD не исчерпан. Эта сумма
не является подтверждённым billing charge. Дополнительных платных запросов
при локальном завершении — 0. Аудит бесплатного завершения находится в
`/tmp/top3-text-local-completion-20261007`.

Bot service и daily timer active; следующий запуск — 2026-10-08 07:30 UTC.
Свежий daily процесс и ручной запуск загружают generator v9. Долго работающий
bot process ещё не перезапущен: `sudo -n systemctl restart top3-news-bot.service`
вернул `sudo: a password is required`. Для загрузки v9 в ручные bot handlers
нужен обычный интерактивный restart; это не мешает следующему daily oneshot.
