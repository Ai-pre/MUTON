#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${PROJECT_ROOT}/run"
LOG_DIR="${PROJECT_ROOT}/logs"
SERVER_PID_FILE="${RUN_DIR}/qwen_server.pid"
CLOUDFLARED_PID_FILE="${RUN_DIR}/cloudflared.pid"
SERVER_LOG="${LOG_DIR}/qwen_server.log"
CLOUDFLARED_LOG="${LOG_DIR}/cloudflared.log"

show_status() {
  local pid_file="$1"
  local name="$2"
  if [ -f "${pid_file}" ] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
    echo "${name}: running (pid $(cat "${pid_file}"))"
  else
    echo "${name}: stopped"
  fi
}

show_status "${SERVER_PID_FILE}" "qwen_server"
show_status "${CLOUDFLARED_PID_FILE}" "cloudflared"

if [ -f "${CLOUDFLARED_LOG}" ]; then
  URL="$(grep -Eo 'https://[-a-z0-9]+\.trycloudflare\.com' "${CLOUDFLARED_LOG}" | tail -n 1 || true)"
  if [ -n "${URL}" ]; then
    echo "tunnel_url: ${URL}"
  fi
fi

echo "server_log: ${SERVER_LOG}"
echo "cloudflared_log: ${CLOUDFLARED_LOG}"
