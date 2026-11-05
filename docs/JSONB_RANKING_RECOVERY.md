# Восстановление ranking после PostgreSQL JSONB

Инцидент 2026-11-05: workflow 159 / ranking 265. Фоновый primary response
`resp_0502aca31cf4eb4a006aec307df14c87d28a4a5cf07fdebb56` создан в 07:30:05 UTC.
Локальный 30-минутный deadline истёк в 08:00:10; ответ завершился в 08:03:28.
Восстановление в 08:00:15 ошибочно остановилось на сравнении prepared request.

Причина: PostgreSQL JSONB меняет порядок ключей вложенных объектов
`published_history`. В snapshot сохраняется и структура selection, и точная
исходная строка input_text. После восстановления build_request выдавал ту же
информацию с другим порядком ключей. Равенство строк ошибочно считало запросы
разными. Обычный Python JSON round trip этого не воспроизводил; нужен PostgreSQL.

`same_event_ranking_request` сравнивает модель и инструкции точно, а input JSON —
после канонизации ключей. Массивы, строки, значения, типы bool/int/float, история
и состав кандидатов сохраняют строгую проверку. Повторяющиеся ключи, невалидный
JSON и NaN отклоняются. Исходный input_text не переписывается: он нужен для прежнего
request key и checkpoint. Версии evaluator v11 / prompt v8 не меняются, поскольку
смысл запроса и бизнес-правила не изменены.

Recovery дополнительно допускает уже упавший ValueError только с точным старым
сообщением, текущей версией evaluator, сохранённым snapshot и доказанным
семантическим равенством при различии строк. Другие ValueError и изменённые
запросы остаются заблокированы. Старые предохранители сохраняются: нет batch/post,
нет завершённого workflow, и первый API POST при восстановлении запрещён.

Проверки выполняются только на cloud-001. `test_ranking_jsonb_recovery` использует
отдельную UTF-8 PostgreSQL 55432/top3_recovery_test и копии двух incident-строк.
Проверяются JSONB reorder, передача исходного input, отказ при изменённых данных,
допуск только конкретного incident failure и откат тестовых изменений. Дополнительно
проходят background transport, frozen history, evaluator, request key и wire schema.
Платных запросов эти тесты не выполняют.

`scripts/resume_saved_daily_workflow.py` продолжает существующий workflow,
сохраняет state-before/result и отдельный cost ledger. Первый primary POST
дополнительно запрещён через require-existing-response. Дополнительные text,
verification и image вызовы ограничиваются заданным `--max-usd` (не более $1),
SDK retry отключён. Уже оплаченный primary читается по прежнему response ID;
он не запускается повторно. Доставка готового выпуска следует прежнему daily
review-процессу и защите от повторной отправки в Telegram.

Для этой правки миграции, зависимости и systemd units не меняются. После проверки
из GitHub код обновляется fast-forward; новый daily/recovery Python-процесс
загружает исправление сразу. Полный перезапуск Telegram polling для исправления
этого CLI ranking пути не требуется.

## Завершённое восстановление 2026-11-05

Шесть проверок прошли на cloud-001, включая PostgreSQL JSONB round trip и
reopen копий incident-строк. Production обновлён из GitHub; исходный response ID
прочитан повторно через GET, его checkpoint теперь completed. Ranking 265 завершён:
91 кандидат, 72 scored records, 27 eligible; сохранённая комбинация 108296.

Workflow 159 / batch 176 / post 180 восстановлены. Текст — 960 символов.
Основной Image API запрос отклонён output-модерацией; резервная платная попытка
не отправлялась, потому что консервативный allowance превысил бы $1.
Использован существующий `complete_budget_retry_with_local_image.py`:
локальная универсальная PNG, image_generation_id=182, SHA-256 проверен.
Пост доставлен редактору в 09:02:54 UTC, review_delivery_attempt_id=92,
telegram_message_id=544; workflow/post/batch awaiting_review. Публикации в каналы
при восстановлении не выполнялись. Таймер 12:00 UTC активен.

Дополнительные подтверждённые расходы Responses: $0.07378700.
Резерв за отклонённый image call: $0.592350 (usage не получен, резерв не снимается).
Всего spent/reserved $0.66613700 из $1. Первый из трёх дополнительных Responses
был узким story-cluster verifier, не повтором основного ranking.
Ledger/state-before: `/tmp/top3-recovery-20261105`; завершение локальной картинкой:
`/tmp/top3-recovery-local-image-20261105`. Изолированная тестовая PostgreSQL
остановлена, pgdata и schema snapshot удалены.
