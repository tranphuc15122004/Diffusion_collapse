#!/usr/bin/env bash
set -euo pipefail

SESSION="${TMUX_SESSION:-diffusion_collapse}"
WORKDIR="/home/tuantb/diffusion_collapse"
LOG_DIR="${WORKDIR}/logs"
mkdir -p "${LOG_DIR}"

if [ $# -lt 2 ]; then
  echo "Usage: $0 <job_name> <command...>"
  exit 1
fi

JOB_NAME="$1"
shift
CMD="$*"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${JOB_NAME}_${TS}.log"
DONE_FILE="${LOG_DIR}/${JOB_NAME}.exit"

rm -f "${DONE_FILE}"

if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SESSION}" -c "${WORKDIR}" -n "${JOB_NAME}"
  TARGET="${SESSION}:${JOB_NAME}"
else
  tmux new-window -t "${SESSION}" -c "${WORKDIR}" -n "${JOB_NAME}"
  TARGET="${SESSION}:${JOB_NAME}"
fi

# Run command inside tmux window, tee output to log file, and write exit code marker
tmux send-keys -t "${TARGET}" \
  "conda activate base 2>/dev/null || true; cd '${WORKDIR}' && (${CMD}) 2>&1 | tee '${LOG_FILE}'; echo \${PIPESTATUS[0]} > '${DONE_FILE}'" C-m

echo "Launched in tmux target '${TARGET}'"
echo "Log file: ${LOG_FILE}"
echo "Exit marker: ${DONE_FILE}"
