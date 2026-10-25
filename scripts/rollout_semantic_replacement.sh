#!/usr/bin/env bash
# Установка только после cloud-001 проверок из GitHub; платных запросов нет.
set -Eeuo pipefail
EXPECTED_COMMIT="${1:?Передай проверенный полный commit SHA}"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
cd /opt/top3-news-bot
sudo -v
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
[[ "$(systemctl show top3-news-daily.service -p ActiveState --value)" == inactive ]]
[[ "$(systemctl show top3-news-auto-publish.service -p ActiveState --value)" != active ]]
git fetch origin main
[[ "$(git rev-parse origin/main)" == "$EXPECTED_COMMIT" ]]
git merge-base --is-ancestor HEAD "$EXPECTED_COMMIT"
umask 077
BACKUP_DIR="$(mktemp -d /tmp/top3-semantic-rollout.XXXXXX)"
cp .env "$BACKUP_DIR/production.env"
git rev-parse HEAD > "$BACKUP_DIR/previous-commit.txt"
git show "$EXPECTED_COMMIT:migrations/021_editorial_replacements.sql" > "$BACKUP_DIR/021.sql"
DB_NAME="$(.venv/bin/python -c 'from app.config import get_settings; print(get_settings().db_name)')"
APPLIED="$(sudo -u postgres psql -X -v ON_ERROR_STOP=1 -d "$DB_NAME" -Atc "SELECT count(*) FROM top3_news.schema_migrations WHERE version='021'")"
if [[ "$APPLIED" == 0 ]]; then
 sudo -u postgres psql -X -v ON_ERROR_STOP=1 -d "$DB_NAME" < "$BACKUP_DIR/021.sql"
fi
git merge --ff-only "$EXPECTED_COMMIT"
export PATH="$HOME/.local/bin:$PATH"
bash scripts/deploy_server.sh
.venv/bin/python - <<'PY'
import asyncio
from app.config import get_settings
from app.db.pool import create_database_pool, close_database_pool
from app.ranking.event_evaluator import EVENT_EVALUATOR_VERSION, EVENT_PROMPT_VERSION
from app.generation.editorial_intent import replacement_positions
assert EVENT_EVALUATOR_VERSION=='event_ranking_evaluator_v11'
assert EVENT_PROMPT_VERSION=='movie_news_event_ranking_prompt_v8'
assert replacement_positions('Замени третью новость. Эта новость уже была вчера.')==(3,)
async def check():
 pool=await create_database_pool(get_settings())
 try:
  assert await pool.fetchval("SELECT EXISTS(SELECT 1 FROM schema_migrations WHERE version='021')")
 finally: await close_database_pool(pool)
asyncio.run(check())
print('Verified:', EVENT_EVALUATOR_VERSION, EVENT_PROMPT_VERSION, 'replace_news', 'migration 021')
PY
.venv/bin/python -m scripts.run_scheduled_auto_publication --dry-run
systemctl is-active top3-news-bot.service top3-news-daily.timer top3-news-auto-publish.timer
systemctl list-timers top3-news-daily.timer top3-news-auto-publish.timer --no-pager --full
printf 'Semantic/replacement rollout verified: %s\nBackup: %s\n' "$EXPECTED_COMMIT" "$BACKUP_DIR"
