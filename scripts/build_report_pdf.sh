#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "${ROOT}/scripts/render_markdown_pdf.py"
ls -lh "${ROOT}"/docs/R5_PILOT_FULL_REPORT.{html,pdf}
