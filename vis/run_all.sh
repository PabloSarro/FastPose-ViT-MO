#!/usr/bin/env bash
# Regenerate every figure in vis/figures (about 2 minutes in dev; reads the BOP tree and results).
#   bash vis/run_all.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
py=../.venv/bin/python
for script in objects scene crops augmentations reprojection stats speed; do
    echo "== $script"
    "$py" "$script.py"
done
