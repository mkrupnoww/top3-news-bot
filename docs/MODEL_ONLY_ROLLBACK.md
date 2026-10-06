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
На момент подготовки этого документа production rollout ещё не выполнен.
