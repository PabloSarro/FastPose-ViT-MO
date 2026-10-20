#!/usr/bin/env bash
# Build a standard BOP dataset tree (as bop_toolkit expects) out of symlinks to the
# nested layout in $DATA_DIR/BOP_Datasets. Nothing is written inside $DATA_DIR.
#
#   $DATA_DIR/BOP_Datasets/lmo/lmo_base/lmo/camera.json  ->  OUT/lmo/camera.json
#   $DATA_DIR/BOP_Datasets/lmo/lmo_models/models_eval    ->  OUT/lmo/models_eval
#   $DATA_DIR/BOP_Datasets/lm/lm_train_pbr/train_pbr     ->  OUT/lm/train_pbr
#   $DATA_DIR/BOP_Datasets/lmo/lmo_test_bop19/test       ->  OUT/lmo/test
#   $DATA_DIR/BOP_Datasets/lmo/lmo_test_all/test         ->  OUT/lmo/test_all  (name taken by test_bop19)
#
# Usage: bash scripts/make_bop_tree.sh [DATASET ...]       (default: lmo lm)
# Then:  export BOP_PATH=$NAS_HOME/datasets/bop
# Re-running is safe: existing links are replaced, other files are left alone.
set -euo pipefail

SRC=${DATA_DIR:?DATA_DIR is not set}/BOP_Datasets
OUT=${BOP_TREE:-${NAS_HOME:?NAS_HOME is not set}/datasets/bop}
datasets=("$@")
(( ${#datasets[@]} )) || datasets=(lmo lm)

link() {  # link TARGET NAME: create or replace the symlink NAME -> TARGET
    if [[ -e "$2" && ! -L "$2" ]]; then
        echo "  skip $2 (exists and is not a symlink)" >&2
        return
    fi
    ln -sfn "$1" "$2"
    echo "  $(basename "$2") -> ${1#"$SRC"/}"
}

for ds in "${datasets[@]}"; do
    src=$SRC/$ds
    [[ -d "$src" ]] || { echo "No dataset folder $src" >&2; exit 1; }
    mkdir -p "$OUT/$ds"
    echo "$ds:"
    declare -A taken=()
    # Top-level files (camera.json, test_targets_*.json, ...) from <ds>_base/<ds>/.
    if [[ -d "$src/${ds}_base/$ds" ]]; then
        for f in "$src/${ds}_base/$ds"/*; do
            link "$f" "$OUT/$ds/$(basename "$f")"
            taken[$(basename "$f")]=1
        done
    fi
    # Split folders from <ds>_<part>/<split>/. test_bop19 goes before test_all, so it gets "test".
    parts=$(cd "$src" && ls -d "${ds}"_*/ 2>/dev/null | sed 's:/$::' | grep -v "^${ds}_base$" \
        | sort -t_ -k2 | awk '/_bop19$/ {print; next} {rest = rest $0 "\n"} END {printf "%s", rest}')
    for part in $parts; do
        for d in "$src/$part"/*/; do
            [[ -d "$d" ]] || continue
            name=$(basename "$d")
            if [[ -n "${taken[$name]:-}" ]]; then
                name=${name}_${part#"${ds}"_}  # e.g. test from lmo_test_all -> test_all
                name=${name/test_test_/test_}
            fi
            link "${d%/}" "$OUT/$ds/$name"
            taken[$name]=1
        done
    done
    unset taken
done
echo "Tree in $OUT. Use: export BOP_PATH=$OUT"
