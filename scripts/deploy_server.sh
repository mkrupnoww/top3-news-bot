#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="/opt/top3-news-bot"
BRANCH="main"

BOT_SERVICE_NAME="top3-news-bot.service"

TIMER_NAMES=(
    "top3-news-collector.timer"
    "top3-news-cleanup.timer"
)

UNIT_NAMES=(
    "top3-news-bot.service"
    "top3-news-collector.service"
    "top3-news-collector.timer"
    "top3-news-cleanup.service"
    "top3-news-cleanup.timer"
)

SYSTEMD_SOURCE_DIR="${PROJECT_DIR}/config/systemd"
SYSTEMD_TARGET_DIR="/etc/systemd/system"

log() {
    printf '[deploy] %s\n' "$*"
}

fail() {
    printf '[deploy] ERROR: %s\n' "$*" >&2
    exit 1
}

show_unit_failure() {
    local unit_name="$1"

    sudo systemctl status \
        "${unit_name}" \
        --no-pager \
        -l \
        || true

    sudo journalctl \
        -u "${unit_name}" \
        -n 50 \
        --no-pager \
        -l \
        || true
}

cd "${PROJECT_DIR}"

log "Checking server working tree"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    git status --short
    fail "Tracked files contain local changes. Deployment stopped."
fi

[[ -f ".env" ]] \
    || fail ".env file is missing"

[[ -x ".venv/bin/python" ]] \
    || fail ".venv Python interpreter is missing"

[[ -f "uv.lock" ]] \
    || fail "uv.lock file is missing"

for unit_name in "${UNIT_NAMES[@]}"; do
    unit_source="${SYSTEMD_SOURCE_DIR}/${unit_name}"

    [[ -f "${unit_source}" ]] \
        || fail "systemd unit file is missing: ${unit_source}"
done

command -v uv >/dev/null 2>&1 \
    || fail "uv command is not available"

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

systemd_changed=false

for unit_name in "${UNIT_NAMES[@]}"; do
    unit_source="${SYSTEMD_SOURCE_DIR}/${unit_name}"
    unit_target="${SYSTEMD_TARGET_DIR}/${unit_name}"

    if [[ ! -f "${unit_target}" ]] \
        || ! cmp -s "${unit_source}" "${unit_target}"; then
        log "Installing updated systemd unit: ${unit_name}"

        sudo install \
            -o root \
            -g root \
            -m 644 \
            "${unit_source}" \
            "${unit_target}"

        systemd_changed=true
    else
        log "systemd unit is already up to date: ${unit_name}"
    fi
done

if [[ "${systemd_changed}" == true ]]; then
    log "Reloading systemd configuration"
    sudo systemctl daemon-reload
fi

log "Restarting ${BOT_SERVICE_NAME}"
sudo systemctl restart "${BOT_SERVICE_NAME}"

if ! sudo systemctl is-active --quiet "${BOT_SERVICE_NAME}"; then
    show_unit_failure "${BOT_SERVICE_NAME}"
    fail "Bot service failed to start"
fi

log "Enabling scheduled timers"
sudo systemctl enable "${TIMER_NAMES[@]}"

log "Restarting scheduled timers"
sudo systemctl restart "${TIMER_NAMES[@]}"

for timer_name in "${TIMER_NAMES[@]}"; do
    if ! sudo systemctl is-active --quiet "${timer_name}"; then
        show_unit_failure "${timer_name}"
        fail "Timer failed to start: ${timer_name}"
    fi
done

CURRENT_COMMIT="$(git rev-parse --short HEAD)"

log "Deployment completed successfully"
log "Commit: ${CURRENT_COMMIT}"
log "Bot service: active"
log "Collector timer: active"
log "Cleanup timer: active"