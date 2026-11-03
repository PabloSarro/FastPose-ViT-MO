#!/usr/bin/env bash
# M7 crop-source comparison with real detections: for each model folder, LM-O test with the BOP
# default detections (GDRNPPDet-PBR) through bop_evaluate.py --detections, then the official BOP19
# eval. Output in <model folder>/eval_det/ (as for tok4x); a folder with eval_det/DONE is skipped.
#
# Usage (from the project root): bash scripts/m7_det_eval.sh MODEL_DIR...
#   MODEL_DIR = a run's variant folder with last.pt, e.g. .../runs/lmo-m7-bobj2x-001/bobj2x
set -euo pipefail
(( $# )) || { sed -n '6,7p' "$0" >&2; exit 5; }
DET=$(ls "$NAS_HOME"/datasets/bop_detections/default_task1/bop23_default_detections_for_task1/gdrnppdet-pbr/gdrnppdet-pbr_lmo-test_*.json)
VIT=${VIT:-vit_b_16_384}
export PYTHONPATH=.
for d in "$@"; do
    out=$d/eval_det
    [[ -e $out/DONE ]] && { echo "== $d: already done"; continue; }
    echo "== $d: predict ($(date '+%F %T'))"
    .venv/bin/python src/bop_evaluate.py --model_weights "$d/last.pt" --splits test \
        --vit_model "$VIT" --depth_mode site --object_conditioning token \
        --detections "$DET" --out_dir "$out"
    echo "== $d: official eval ($(date '+%F %T'))"
    bash scripts/bop_eval.sh "$out" fastposevitmo_lmo-test.csv > "$out/bop_eval_test.log" 2>&1
    touch "$out/DONE"
done
echo "== all done ($(date '+%F %T'))"
