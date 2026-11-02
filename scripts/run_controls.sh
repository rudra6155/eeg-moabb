#!/usr/bin/env bash
# Label-shuffle negative control. `--smoke` runs 2 subjects into results_smoke/.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

"$PY" -m eeg.controls "$@"
