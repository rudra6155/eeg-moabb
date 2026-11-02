#!/usr/bin/env bash
# Both pipelines x both protocols (run-grouped headline, cross-session secondary).
# `--smoke` runs 2 subjects into results_smoke/ and asserts both pipelines beat chance.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

"$PY" -m eeg.evaluate "$@"
