#!/usr/bin/env bash
# Paper (docs/PAPER_TODO.md §2, first item): every ablation model on one metric, the official BOP19
# AR on LM-O test with the BOP default detections (GDRNPPDet-PBR), evaluating last.pt as always.
# Output in <run>/<variant>/eval_det/ (as M7); a folder with eval_det/DONE is skipped, so models
# already evaluated (tok4x, M5, M7) are not redone.
# The M3 models are driller-only (obj 8): their CSV holds only the driller and the official eval
# uses a driller-only targets file (scripts/bop_subset_targets.py), so their AR is a driller AR,
# comparable among the M3 models only.
#
# Usage (from the project root, in dev or a job): bash scripts/paper_ablation_eval.sh [NAME...]
#   no NAME = all models of the list below
set -euo pipefail
export PYTHONPATH=.
RUNS=$NAS_HOME/outputs/FastPose-ViT/runs
DET=$(ls "$NAS_HOME"/datasets/bop_detections/default_task1/bop23_default_detections_for_task1/gdrnppdet-pbr/gdrnppdet-pbr_lmo-test_*.json)
TARGETS8=test_targets_bop19_obj8.json

# name -> "run/variant conditioning depth_mode obj_ids"
declare -A MODELS=(
    [m3-base]="lmo-m3-abl-base-001/base heads site 8"
    [m3-base2]="lmo-m3-abl-base2-001/base2 heads site 8"
    [m3-legacy]="lmo-m3-abl-legacy-001/legacy heads legacy 8"
    [m3-dec]="lmo-m3-abl-dec-001/dec heads site 8"
    [m3-domgap]="lmo-m3-abl-domgap-001/domgap heads site 8"
    [std]="lmo-m4-std-001/std heads site all"
    [std2]="lmo-m4-std2-001/std2 heads site all"
    [dg]="lmo-m4-dg-001/dg heads site all"
    [dg2]="lmo-m4-dg2-001/dg2 heads site all"
    [dg2x]="lmo-m4-dg2x-001/dg2x heads site all"
    [dg2x2]="lmo-m4-dg2x2-001/dg2x2 heads site all"
    [dg4x]="lmo-m4-dg4x-001/dg4x heads site all"
    [tok2x]="lmo-m4-tok2x-001/tok2x token site all"
    [tok2x2]="lmo-m4-tok2x2-001/tok2x2 token site all"
    [both2x]="lmo-m4-both2x-001/both2x both site all"
    [both2x2]="lmo-m4-both2x2-001/both2x2 both site all"
    [tok4x2]="lmo-m4-tok4x2-001/tok4x2 token site all"
)
ORDER=(m3-base m3-base2 m3-legacy m3-dec m3-domgap std std2 dg dg2 dg2x dg2x2 dg4x tok2x tok2x2
       both2x both2x2 tok4x2)
(( $# )) && ORDER=("$@")

if [[ ! -e $NAS_HOME/datasets/bop/lmo/$TARGETS8 ]]; then
    .venv/bin/python scripts/bop_subset_targets.py --dataset lmo --obj_ids 8 --name "$TARGETS8"
fi
for name in "${ORDER[@]}"; do
    [[ -n ${MODELS[$name]:-} ]] || { echo "unknown model $name"; exit 5; }
    read -r path cond depth objs <<< "${MODELS[$name]}"
    out=$RUNS/$path/eval_det
    [[ -e $out/DONE ]] && { echo "== $name: already done"; continue; }
    obj_args=() eval_args=()
    if [[ $objs != all ]]; then
        obj_args=(--obj_ids "$objs")
        eval_args=(--targets_filename="$TARGETS8")
    fi
    echo "== $name: predict ($(date '+%F %T'))"
    .venv/bin/python src/bop_evaluate.py --dataset lmo --model_weights "$RUNS/$path/last.pt" \
        --splits test --depth_mode "$depth" --object_conditioning "$cond" "${obj_args[@]}" \
        --detections "$DET" --out_dir "$out"
    echo "== $name: official eval ($(date '+%F %T'))"
    bash scripts/bop_eval.sh "$out" fastposevitmo_lmo-test.csv "${eval_args[@]}" \
        > "$out/bop_eval_test.log" 2>&1
    touch "$out/DONE"
done
echo "== all done ($(date '+%F %T'))"
