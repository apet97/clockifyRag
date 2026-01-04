#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="m1_compatibility.log"
: > "$LOG_FILE"

{
  echo "M1 compatibility check"
  echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "OS: $(uname -a)"

  machine="$(python3 - <<'PY'
import platform
print(platform.machine().lower())
PY
)"
  echo "Machine: ${machine}"
  if [[ "${machine}" != "arm64" && "${machine}" != "aarch64" ]]; then
    echo "Expected Apple Silicon (arm64/aarch64), got: ${machine}" >&2
    exit 1
  fi

  python3 - <<'PY'
import sys
import platform
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
PY

  python3 - <<'PY'
try:
    import numpy as np
    print(f"NumPy: {np.__version__}")
except Exception as exc:
    print(f"NumPy: not installed ({exc})")
PY
} | tee -a "$LOG_FILE"
