#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="/opt/top3-news-bot"
BRANCH="main"
SERVICE_NAME="top3-news-bot.service"

UNIT_SOURCE="${PROJECT_DIR}/config/systemd/top3-news-bot.service"
UNIT_TARGET="/etc/systemd/system/top3-news-bot.service"

log() {
    printf '[deploy] %s\n' "$*"
}

fail() {
    printf '[deploy] ERROR: %s\n' "$*" >&2
    exit 1
}

cd "${PROJECT_DIR}"

log "Checking server working tree"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    git status --short
    fail "Tracked files contain local changes. Deployment stopped."
fi

[[ -f ".env" ]] || fail ".env file is missing"
[[ -x ".venv/bin/python" ]] || fail ".venv Python interpreter is missing"
[[ -f "uv.lock" ]] || fail "uv.lock file is missing"
[[ -f "${UNIT_SOURCE}" ]] || fail "systemd unit file is missing"

command -v uv >/dev/null 2>&1 || fail "uv command is not available"

OLD_COMMIT="$(git rev-parse HEAD)"

log "Fetching origin/${BRANCH}"
git fetch origin "${BRANCH}"

NEW_COMMIT="$(git rev-parse "origin/${BRANCH}")"

if [[ "${OLD_COMMIT}" == "${NEW_COMMIT}" ]]; then
    log "Repository is already up to date"
else
    log "Updating ${OLD_COMMIT:0:7} -> ${NEW_COMMIT:0:7}"
    git merge --ff-only "origin/${BRANCH}"
fi

log "Synchronizing Python dependencies"
uv sync --frozen --no-dev

log "Checking Python syntax"
.venv/bin/python -m compileall -q app scripts

if [[ ! -f "${UNIT_TARGET}" ]] || ! cmp -s "${UNIT_SOURCE}" "${UNIT_TARGET}"; then
    log "Installing updated systemd unit"
    sudo install \
        -o root \
        -g root \
        -m 644 \
        "${UNIT_SOURCE}" \
        "${UNIT_TARGET}"

    sudo systemctl daemon-reload
else
    log "systemd unit is already up to date"
fi

log "Restarting ${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"

if ! sudo systemctl is-active --quiet "${SERVICE_NAME}"; then
    sudo systemctl status "${SERVICE_NAME}" --no-pager -l || true
    sudo journalctl -u "${SERVICE_NAME}" -n 50 --no-pager -l || true
    fail "Service failed to start"
fi

CURRENT_COMMIT="$(git rev-parse --short HEAD)"

log "Deployment completed successfully"
log "Commit: ${CURRENT_COMMIT}"
log "Service: active"