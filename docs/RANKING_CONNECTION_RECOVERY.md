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
