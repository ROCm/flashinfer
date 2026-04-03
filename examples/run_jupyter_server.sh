#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Start JupyterLab for amd_flashinfer_rocm_tutorial.ipynb (or any notebook in the repo).
#
# Usage (activate your ROCm/flashinfer env first, e.g. inside rocm/flashinfer Docker):
#   cd /path/to/flashinfer
#   ./examples/run_jupyter_server.sh
#
# Remote / SSH: listen on all interfaces (default) and forward the port:
#   ssh -L 8888:localhost:8888 user@node
# Then open the printed http://127.0.0.1:8888/lab?token=... in a browser.
#
# Override port: JUPYTER_PORT=8890 ./examples/run_jupyter_server.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! python -c "import jupyterlab" 2>/dev/null; then
  echo "Installing jupyterlab into the current Python environment..."
  pip install jupyterlab
fi

PORT="${JUPYTER_PORT:-8888}"
IP="${JUPYTER_IP:-0.0.0.0}"

echo "Starting JupyterLab from: $ROOT"
echo "  URL: http://127.0.0.1:${PORT}/lab (use SSH -L if remote)"
echo "  Stop: Ctrl+C"
echo ""

exec python -m jupyter lab \
  --no-browser \
  --ip="$IP" \
  --port="$PORT" \
  --notebook-dir="$ROOT" \
  "$@"
