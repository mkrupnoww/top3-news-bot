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
