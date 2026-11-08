#!/usr/bin/env bash
# M9 bullet 2: LM-O (or YCB-V) test with the BOP default detections (as M7), the network run
# as TensorRT engines (optimization/bop_trt.py), then the official BOP19 eval. Output in
# ENGINE_DIR/eval_det_<engine name>/; a folder with DONE is skipped.
#
# Usage (from the project root): [DATASET=ycbv] bash scripts/m9_trt_eval.sh ENGINE...
#   e.g. bash scripts/m9_trt_eval.sh $NAS_HOME/results/m9/lmo-tok4x/model_fp16.engine
#   DATASET: lmo (default) or ycbv
set -euo pipefail
(( $# )) || { sed -n '6,7p' "$0" >&2; exit 5; }
DATASET=${DATASET:-lmo}
DET=$(ls "$NAS_HOME"/datasets/bop_detections/default_task1/bop23_default_detections_for_task1/gdrnppdet-pbr/gdrnppdet-pbr_"$DATASET"-test_*.json)
export PYTHONPATH=.
for engine in "$@"; do
    name=$(basename "$engine" .engine)
    out=$(dirname "$engine")/eval_det_$name
    [[ -e $out/DONE ]] && { echo "== $name: already done"; continue; }
    echo "== $name: predict ($(date '+%F %T'))"
    .venv/bin/python src/bop_evaluate.py --dataset "$DATASET" --backend trt \
        --trt_engine "$engine" --splits test \
        --depth_mode site --object_conditioning token --batch_size 16 \
        --detections "$DET" --out_dir "$out"
    echo "== $name: official eval ($(date '+%F %T'))"
    bash scripts/bop_eval.sh "$out" fastposevitmo_"$DATASET"-test.csv > "$out/bop_eval_test.log" 2>&1
    touch "$out/DONE"
done
echo "== all done ($(date '+%F %T'))"
