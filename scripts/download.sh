#!/usr/bin/env bash
# Fetch and cache BNCI2014_001 into ./data and write results/dataset_manifest.json.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

"$PY" -m eeg.data
