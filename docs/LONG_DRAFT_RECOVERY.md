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
