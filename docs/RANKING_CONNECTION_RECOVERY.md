# Восстановление ранжирования после обрыва 2026-10-13

Утренний workflow 89 / ranking 241 стартовал в 07:30 UTC и упал в 07:37.
77 кандидатов, 0 обработанных; text/image/Telegram ещё не начинались.
Traceback: чтение HTTP response body -> httpcore.ReadError -> httpx.ReadError
-> openai.APIConnectionError. Это не APITimeoutError и не отказ по ключу.
Бесплатный models.retrieve подтвердил доступ к gpt-6.1-sol; SDK использует
api.openai.com, proxy environment отсутствует. Источник исторического обрыва
на уровне сети/провайдера без серверной телеметрии установить нельзя.

Причина потери выпуска в приложении: foreground ranking без сохранённого
response ID и восстановления чтения; ошибка немедленно делает workflow failed.
store=false исключал последующее получение ответа утреннего запроса.
Его usage/billing неизвестны; нулевую стоимость утверждать нельзя.

Event ranking v9 выполняет один background POST (SDK retries=0), сохраняет
response ID в data/openai_event_responses и опрашивает GET каждые 10 секунд.
Transient GET errors повторяют только чтение, без новой генерации, до 30 минут.
Checkpoint сохраняется до POST; при неизвестном исходе POST новая генерация
не запускается автоматически. Completed output/usage сохраняются атомарно;
повторное чтение той же операции использует checkpoint. Failed/incomplete/
cancelled не принимаются за успешное ранжирование. Один файловый lock запрещает
одновременное выполнение одного запроса двумя процессами.

Основание: [официальная документация Background mode](https://developers.openai.com/api/docs/guides/background).
Фоновое выполнение и GET-проверки устраняют зависимость длительной генерации
от одного HTTP-соединения. Store=true сохраняет ответ для восстановления.
Это не гарантия доступности внешнего API при длительной аварии.

Явный однократный retry старого failed ranking:
`python -m scripts.retry_failed_ranking_workflow --workflow-id 89
--ledger-directory /tmp/top3-ranking-retry-20261013 --max-usd 1`.
Сначала применяется --check-only. Исходный failed ranking остаётся в БД,
дата и cutoff сохраняются, новый evaluator version создаёт новый request key.
Один общий ledger ограничивает ranking/text/self-review/image до $1.
Manual ranking output limit 30000 tokens, text 5000, reasoning low,
search max_tool_calls=1; queued response allowance освобождается только после
получения completed usage. Unknown errors сохраняют резерв.

## Автоматическое продолжение того же ranking run

До вызова модели в ranking_runs.parameters сохраняется background_snapshot:
полный состав кандидатов, даты/веса/метрики и точный EventRankingModelRequest.
После interrupted GET или остановки процесса daily workflow может вернуть
незавершённый background ranking в running и продолжить тот же ranking ID/key.
Кандидаты заново не выбираются. Первый POST при recovery запрещён: обязателен
checkpoint с response ID. Неизвестный исход POST остаётся blocked, новый запрос
не создаётся. История ошибок сохраняется в background_recoveries.
Последующие ещё не начатые проверки/исправления результата следуют обычному
evaluator; completed ответы читаются из cache. Завершённый daily workflow
возвращается до создания SDK runtimes и Telegram Bot.

Scheduled runner использует saved as_of при повторном входе в тот же день.
Пользователь согласовал две дополнительные проверки в 07:45 и 08:00 UTC.
Их условия и установка описаны в DAILY_RECOVERY_TIMER.md; проверки до конца
дня не используются. Для активации нового systemd расписания нужен обычный deploy.

## Проверенный результат 2026-10-13

Повторный ranking 242 completed (основное ранжирование + global story-cluster
verification); TOP-3 combination 75007, news IDs 7072/7079/7068.
Post 154 / batch 153 содержит 849 символов и прошёл self-review / repair.
Первый Image API запрос отклонён moderation; image 155 failed. Запрос не вернул
usage, поэтому conservative allowance 0.592645 USD сохранён. Следующая попытка
image 156 остановлена бюджетом ДО API: accounted 0.92787150 USD + allowance
0.499090 USD не помещались в разрешённый $1.

Штатный local PNG fallback завершил image 157 без дополнительного OpenAI call.
Workflow 89 и post 154 — awaiting_review. Review delivery 69: sent, message 464.
Это доставка редактору на проверку, не обход обычного approval/publication flow.
После подтверждения редактором в 08:53:10 UTC post 154 имеет status published:
publication attempt 72, channel message 3605. Проверено по БД и журналу бота.
Исходный ranking 241 остаётся failed. Дата и cutoff 07:30:05 UTC сохранены.

Ledger /tmp/top3-ranking-retry-20261013/cost-ledger.json: пять completed Responses
вызовов (ranking, verifier, text, self-review, repair) с суммарной оценкой
**0.33522650 USD**, один moderation-rejected Image request с неизвестным usage
и резервом **0.592645 USD**. Итого учтено/зарезервировано **0.92787150 USD**.
Это не billing invoice; стоимость утреннего оборванного автоматического запроса
неизвестна и не объявляется нулевой.

На cloud-001 прошли 8 offline scripts, бюджетная queued/GET проверка и 2 теста
изолированной PostgreSQL 55432: manual recovery history/duplicate blocking и
полное восстановление background ranking с тем же ID/key/frozen request до
completion. Compileall и systemd unit verification прошли. Production данные
тестами БД не менялись; PostgreSQL 55432 остановлен, snapshot/pgdata удалены.
Отдельно на production проверен повторный вход scheduled runner в завершённый
workflow с запрещёнными SDK factories и Telegram Bot: завершился без API
runtime, новой доставки и повторной публикации, сохранив исходный cutoff.
