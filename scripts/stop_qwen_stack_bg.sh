#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${PROJECT_ROOT}/run"
SERVER_PID_FILE="${RUN_DIR}/qwen_server.pid"
CLOUDFLARED_PID_FILE="${RUN_DIR}/cloudflared.pid"

stop_pid() {
  local pid_file="$1"
  local name="$2"
  if [ ! -f "${pid_file}" ]; then
    echo "${name}: no pid file"
    return
  fi

  local pid
  pid="$(cat "${pid_file}")"
  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    sleep 1
    if kill -0 "${pid}" 2>/dev/null; then
      kill -9 "${pid}" 2>/dev/null || true
    fi
    echo "${name}: stopped pid ${pid}"
  else
    echo "${name}: pid ${pid} not running"
  fi
  rm -f "${pid_file}"
}

stop_pid "${CLOUDFLARED_PID_FILE}" "cloudflared"
stop_pid "${SERVER_PID_FILE}" "qwen_server"
