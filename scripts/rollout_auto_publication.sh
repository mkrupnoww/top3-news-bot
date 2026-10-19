#!/usr/bin/env bash
# Запуск на cloud-001 после проверки GitHub snapshot в изолированной БД.
set -Eeuo pipefail
EXPECTED_COMMIT="${1:?Передай проверенный полный commit SHA}"
CHANNEL_CONFIG="${2:-/tmp/top3-auto-publish-channels.json}"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
cd /opt/top3-news-bot
sudo -v
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
[[ "$(systemctl show top3-news-daily.service -p ActiveState --value)" == inactive ]]
[[ "$(systemctl show top3-news-auto-publish.service -p ActiveState --value)" != active ]]
[[ -f "$CHANNEL_CONFIG" ]]
git fetch origin main
[[ "$(git rev-parse origin/main)" == "$EXPECTED_COMMIT" ]]
git merge-base --is-ancestor HEAD "$EXPECTED_COMMIT"
umask 077
BACKUP_DIR="$(mktemp -d /tmp/top3-auto-publication-rollout.XXXXXX)"
cp .env "$BACKUP_DIR/production.env"
git rev-parse HEAD > "$BACKUP_DIR/previous-commit.txt"
git show "$EXPECTED_COMMIT:migrations/020_daily_auto_publications.sql" > "$BACKUP_DIR/020.sql"
DB_NAME="$(.venv/bin/python -c 'from app.config import get_settings; print(get_settings().db_name)')"
APPLIED="$(sudo -u postgres psql -X -v ON_ERROR_STOP=1 -d "$DB_NAME" -Atc "SELECT count(*) FROM top3_news.schema_migrations WHERE version='020'")"
if [[ "$APPLIED" == 0 ]]; then
    # postgres читает stdin: закрытый mktemp-каталог не нужно открывать другим пользователям.
    sudo -u postgres psql -X -v ON_ERROR_STOP=1 -d "$DB_NAME" < "$BACKUP_DIR/020.sql"
fi
git merge --ff-only "$EXPECTED_COMMIT"
.venv/bin/python - "$CHANNEL_CONFIG" <<'PY'
from pathlib import Path
import json
import re
import sys
from app.config import Settings
values = json.loads(Path(sys.argv[1]).read_text())
assert set(values) == {'CHANNEL_PROD_ID', 'CHANNEL_PROD_VIP_ID'}
assert len(set(values.values())) == 2
assert all(isinstance(v, int) and str(v).startswith('-100') for v in values.values())
Settings(**values, AUTO_PUBLICATION_ENABLED=True)
path = Path('.env')
text = path.read_text()
for key, value in {**values, 'AUTO_PUBLICATION_ENABLED': 'true'}.items():
    pattern = re.compile(r'^' + re.escape(key) + r'=.*$', re.MULTILINE)
    text = pattern.sub(key + '=' + str(value), text) if pattern.search(text) else text.rstrip() + '\n' + key + '=' + str(value) + '\n'
path.write_text(text)
path.chmod(0o600)
print('Два production-канала настроены; TELEGRAM_CHANNEL_ID сохранён')
PY
.venv/bin/python -m scripts.check_auto_publication_channels
export PATH="$HOME/.local/bin:$PATH"
bash scripts/deploy_server.sh
.venv/bin/python -m scripts.run_scheduled_auto_publication --dry-run
systemctl is-active top3-news-bot.service top3-news-daily.timer top3-news-auto-publish.timer
systemctl list-timers top3-news-auto-publish.timer --no-pager --full
printf 'Auto publication rollout verified: %s\nBackup: %s\n' "$EXPECTED_COMMIT" "$BACKUP_DIR"
