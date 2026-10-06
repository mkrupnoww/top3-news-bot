#!/usr/bin/env bash
# cloud-002 -> GitHub -> cloud-001: код 3309878 с обновлёнными моделями.
set -Eeuo pipefail

EXPECTED_COMMIT="${1:?Передай проверенный полный commit SHA}"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]]
cd /opt/top3-news-bot

if [[ -t 0 ]]; then
    sudo -v
else
    sudo -n true || {
        printf 'Для restart требуется интерактивный sudo; production не изменён.\n' >&2
        exit 1
    }
fi

[[ -z "$(git status --porcelain --untracked-files=no)" ]]
[[ "$(systemctl show top3-news-daily.service -p ActiveState --value)" == inactive ]]
git fetch origin main
[[ "$(git rev-parse origin/main)" == "${EXPECTED_COMMIT}" ]]
git merge-base --is-ancestor HEAD "${EXPECTED_COMMIT}"

if ! command -v uv >/dev/null 2>&1 && [[ -x "${HOME}/.local/bin/uv" ]]; then
    export PATH="${HOME}/.local/bin:${PATH}"
fi
command -v uv >/dev/null 2>&1
uv --version
[[ -x .venv/bin/python && -f .env ]]
[[ -f data/images/versatile_option/versatile_option.png ]]
# Lockfile не изменён относительно production; dry-run проверяет зависимости.
[[ "$(git rev-parse HEAD:uv.lock)" == "$(git rev-parse "${EXPECTED_COMMIT}:uv.lock")" ]]
uv sync --frozen --no-dev --dry-run

umask 077
BACKUP_DIR="$(mktemp -d /tmp/top3-model-only-rollout.XXXXXX)"
cp .env "${BACKUP_DIR}/production.env"
git rev-parse HEAD > "${BACKUP_DIR}/previous-commit.txt"
git show "${EXPECTED_COMMIT}:scripts/deploy_server.sh" > "${BACKUP_DIR}/deploy_server.sh"
bash -n "${BACKUP_DIR}/deploy_server.sh"
printf 'Backup: %s\n' "${BACKUP_DIR}"

.venv/bin/python - <<'PY'
from pathlib import Path
import re

path = Path('.env')
text = path.read_text()
for key in ('OPENAI_IMAGE_PROMPT_AGENT_MODEL', 'OPENAI_IMAGE_QUALITY'):
    text = re.sub(r'^' + key + r'=[^\n]*(?:\n|$)', '', text, flags=re.MULTILINE)
for key, value in {
    'OPENAI_RANKING_MODEL': 'gpt-6.1-sol',
    'OPENAI_GENERATION_MODEL': 'gpt-6.1-sol',
    'OPENAI_IMAGE_MODEL': 'gpt-image-2.5-flare',
}.items():
    pattern = re.compile(r'^' + re.escape(key) + r'=.*$', re.MULTILINE)
    text = pattern.sub(key + '=' + value, text) if pattern.search(text) else text.rstrip() + '\n' + key + '=' + value + '\n'
path.write_text(text)
print('Ranking/generation/image models updated; agent settings removed')
PY

# Не применяем и не откатываем миграции; история agent plans остаётся в БД.
bash "${BACKUP_DIR}/deploy_server.sh"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
.venv/bin/python - <<'PY'
from pathlib import Path
from app.config import Settings
s = Settings()
assert s.openai_ranking_model == s.openai_generation_model == 'gpt-6.1-sol'
assert s.openai_image_model == 'gpt-image-2.5-flare'
assert not Path('app/generation/image_prompt_agent.py').exists()
print('Verified models:', s.openai_ranking_model, s.openai_generation_model, s.openai_image_model)
print('Image prompt: baseline direct generator; Image Prompt Agent removed')
PY
systemctl is-active top3-news-bot.service top3-news-collector.timer top3-news-cleanup.timer top3-news-daily.timer
systemctl list-timers top3-news-daily.timer --no-pager
printf 'Model-only rollout verified: %s\nBackup: %s\n' "${EXPECTED_COMMIT}" "${BACKUP_DIR}"
