#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${PROJECT_ROOT}/run"
LOG_DIR="${PROJECT_ROOT}/logs"
URL_WORKTREE="${MUTON_URL_WORKTREE:-$HOME/MUTON_server}"
GIT_REMOTE="${MUTON_GIT_REMOTE:-origin}"
SERVER_BRANCH="${MUTON_SERVER_BRANCH:-server_main}"
URL_BRANCH="${MUTON_URL_BRANCH:-server}"
URL_WORKTREE_BRANCH="${MUTON_URL_WORKTREE_BRANCH:-server-url-runtime}"
CONDA_SH="${MUTON_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${MUTON_CONDA_ENV:-muton}"
SERVER_HOST="${MUTON_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${MUTON_SERVER_PORT:-5000}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
QWEN_ADAPTER="${MUTON_QWEN_ADAPTER:-$PROJECT_ROOT/out/qwen_omni_lora/ko_stage}"
STT_BACKEND="${MUTON_QWEN_STT_BACKEND:-openai}"

SERVER_PID_FILE="${RUN_DIR}/qwen_server.pid"
CLOUDFLARED_PID_FILE="${RUN_DIR}/cloudflared.pid"
SERVER_LOG="${LOG_DIR}/qwen_server.log"
CLOUDFLARED_LOG="${LOG_DIR}/cloudflared.log"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

ensure_repo_updated() {
  git -C "${PROJECT_ROOT}" remote set-url "${GIT_REMOTE}" https://github.com/Ai-pre/MUTON.git >/dev/null 2>&1 || true
  git -C "${PROJECT_ROOT}" fetch "${GIT_REMOTE}" --prune
  git -C "${PROJECT_ROOT}" checkout "${SERVER_BRANCH}"
  git -C "${PROJECT_ROOT}" pull "${GIT_REMOTE}" "${SERVER_BRANCH}"
}

ensure_url_worktree() {
  if [ ! -e "${URL_WORKTREE}" ]; then
    git -C "${PROJECT_ROOT}" worktree add -B "${URL_WORKTREE_BRANCH}" "${URL_WORKTREE}" "${GIT_REMOTE}/${URL_BRANCH}"
  fi
  git -C "${URL_WORKTREE}" remote set-url "${GIT_REMOTE}" https://github.com/Ai-pre/MUTON.git >/dev/null 2>&1 || true
}

is_running() {
  local pid_file="$1"
  [ -f "${pid_file}" ] && kill -0 "$(cat "${pid_file}")" 2>/dev/null
}

start_server() {
  if is_running "${SERVER_PID_FILE}"; then
    echo "Qwen server already running (pid $(cat "${SERVER_PID_FILE}"))"
    return
  fi

  if [ "${STT_BACKEND}" = "openai" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "OPENAI_API_KEY is required when MUTON_QWEN_STT_BACKEND=openai" >&2
    exit 1
  fi

  : > "${SERVER_LOG}"
  nohup bash -lc "
    set -euo pipefail
    cd '${PROJECT_ROOT}'
    source '${CONDA_SH}'
    conda activate '${CONDA_ENV}'
    export MUTON_QWEN_STT_BACKEND='${STT_BACKEND}'
    export MUTON_QWEN_ADAPTER='${QWEN_ADAPTER}'
    export CUDA_VISIBLE_DEVICES='${CUDA_DEVICES}'
    python scripts/run_qwen_server.py
  " >> "${SERVER_LOG}" 2>&1 &
  echo $! > "${SERVER_PID_FILE}"
  echo "Started Qwen server (pid $(cat "${SERVER_PID_FILE}"))"
}

wait_for_server() {
  local tries=60
  for ((i=1; i<=tries; i++)); do
    if curl -fsS "http://${SERVER_HOST}:${SERVER_PORT}/health" >/dev/null 2>&1; then
      echo "Server is healthy at http://${SERVER_HOST}:${SERVER_PORT}/health"
      return
    fi
    sleep 1
  done
  echo "Server health check timed out. See ${SERVER_LOG}" >&2
  exit 1
}

start_cloudflared() {
  if is_running "${CLOUDFLARED_PID_FILE}"; then
    echo "cloudflared already running (pid $(cat "${CLOUDFLARED_PID_FILE}"))"
    return
  fi

  : > "${CLOUDFLARED_LOG}"
  nohup cloudflared tunnel --url "http://${SERVER_HOST}:${SERVER_PORT}" >> "${CLOUDFLARED_LOG}" 2>&1 &
  echo $! > "${CLOUDFLARED_PID_FILE}"
  echo "Started cloudflared (pid $(cat "${CLOUDFLARED_PID_FILE}"))"
}

discover_tunnel_url() {
  local tries=60
  local url=""
  for ((i=1; i<=tries; i++)); do
    url="$(grep -Eo 'https://[-a-z0-9]+\.trycloudflare\.com' "${CLOUDFLARED_LOG}" | tail -n 1 || true)"
    if [ -n "${url}" ]; then
      printf '%s' "${url}"
      return
    fi
    sleep 1
  done
  echo "Could not discover trycloudflare URL. See ${CLOUDFLARED_LOG}" >&2
  exit 1
}

update_backend_url() {
  local url="$1"
  ensure_url_worktree
  git -C "${URL_WORKTREE}" fetch "${GIT_REMOTE}" --prune
  git -C "${URL_WORKTREE}" checkout "${URL_WORKTREE_BRANCH}"
  git -C "${URL_WORKTREE}" rebase "${GIT_REMOTE}/${URL_BRANCH}" || {
    echo "Rebase failed in ${URL_WORKTREE}. Resolve manually." >&2
    exit 1
  }

  (
    cd "${URL_WORKTREE}"
    python scripts/update_backend_url.py "${url}"
    git add backend_url.json
    if git diff --cached --quiet; then
      echo "backend_url.json already up to date"
    else
      git commit -m "Update backend URL"
      git push "${GIT_REMOTE}" HEAD:"${URL_BRANCH}"
    fi
  )
}

main() {
  ensure_repo_updated
  start_server
  wait_for_server
  start_cloudflared
  TUNNEL_URL="$(discover_tunnel_url)"
  update_backend_url "${TUNNEL_URL}"

  echo
  echo "Stack started successfully."
  echo "Tunnel URL: ${TUNNEL_URL}"
  echo "Server log: ${SERVER_LOG}"
  echo "cloudflared log: ${CLOUDFLARED_LOG}"
  echo
  echo "Health checks:"
  echo "  curl http://${SERVER_HOST}:${SERVER_PORT}/health"
  echo "  curl ${TUNNEL_URL}/health"
}

main "$@"
