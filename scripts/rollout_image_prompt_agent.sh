#!/usr/bin/env bash
# Однократный rollout: Git/GitHub -> additive migration -> .env -> штатный deploy.
set -Eeuo pipefail

EXPECTED_COMMIT="${1:?Передай проверенный полный commit SHA}"
PROJECT_DIR="/opt/top3-news-bot"
cd "${PROJECT_DIR}"

if [[ -t 0 ]]; then
    sudo -v
else
    sudo -n true || {
        printf 'Для миграции и restart требуется интерактивный sudo; изменений production не выполнено.\n' >&2
        exit 1
    }
fi

[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
    printf 'Production tracked working tree содержит изменения; rollout остановлен.\n' >&2
    exit 1
}
[[ "$(systemctl is-active top3-news-daily.service || true)" != active ]] || {
    printf 'Daily workflow сейчас выполняется; rollout остановлен.\n' >&2
    exit 1
}
git fetch origin main
[[ "$(git rev-parse origin/main)" == "${EXPECTED_COMMIT}" ]] || {
    printf 'origin/main не совпадает с проверенным commit; rollout остановлен.\n' >&2
    exit 1
}

umask 077
BACKUP_DIR="$(mktemp -d /tmp/top3-image-agent-rollout.XXXXXX)"
cp .env "${BACKUP_DIR}/production.env"
sudo -u postgres pg_dump --schema-only --no-owner --schema=top3_news top3_news_db > "${BACKUP_DIR}/schema-before.sql"
git show "${EXPECTED_COMMIT}:migrations/019_image_prompt_plans.sql" > "${BACKUP_DIR}/019.sql"

APPLIED="$(sudo -u postgres psql -X -A -t -d top3_news_db -c "SELECT EXISTS (SELECT 1 FROM top3_news.schema_migrations WHERE version='019')")"
if [[ "${APPLIED}" == f ]]; then
    # Файл открывает текущий пользователь до sudo: приватная backup directory
    # остаётся 0700, postgres получает SQL через stdin в той же psql session.
    sudo -u postgres psql -X -v ON_ERROR_STOP=1 -d top3_news_db \
        -c 'SET ROLE michael_psql' -f - < "${BACKUP_DIR}/019.sql"
fi

# Environment-specific настройки — разрешённая эксплуатационная конфигурация.
.venv/bin/python - <<'PY'
from pathlib import Path
import re

path = Path('.env')
text = path.read_text()
updates = {
    'OPENAI_IMAGE_MODEL': 'gpt-image-2.5-flare',
    'OPENAI_IMAGE_PROMPT_AGENT_MODEL': 'gpt-6.1-sol',
    'OPENAI_IMAGE_QUALITY': 'medium',
    'OPENAI_IMAGE_SIZE': '1024x1024',
}
for key, value in updates.items():
    pattern = re.compile(r'^' + re.escape(key) + r'=.*$', re.MULTILINE)
    text = pattern.sub(key + '=' + value, text) if pattern.search(text) else text.rstrip() + '\n' + key + '=' + value + '\n'
path.write_text(text)
print('Image environment settings updated; ranking/generation preserved')
PY

bash scripts/deploy_server.sh
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
sudo -u postgres psql -X -d top3_news_db -c \
    "SELECT version, applied_at FROM top3_news.schema_migrations WHERE version='019'"
systemctl list-timers top3-news-daily.timer --no-pager
printf 'Image Prompt Agent rollout verified: %s\nBackup: %s\n' "${EXPECTED_COMMIT}" "${BACKUP_DIR}"
