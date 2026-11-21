#!/usr/bin/env bash
# M19 R2 on the dense-head poses: the adopted refiner (cnxs, seed a) refines the PnP poses of
# a multi-head model (default: lmo-m18-densesym-001). 0 = the PnP poses through the same code.
# Usage (dev or a job, project root): bash scripts/m19_on_dense.sh [DENSE_RUN_NAME VARIANT]
# Output: $NAS_HOME/results/m19_on_dense_<VARIANT>/{eval_valn_r<n>,eval_det_r<n>}
set -euo pipefail
export PYTHONPATH=.
RUNS=$NAS_HOME/outputs/FastPose-ViT/runs
JOB=${1:-lmo-m18-densesym-001} V=${2:-densesym}
DENSE=$RUNS/$JOB/$V/last.pt
REFINER=$RUNS/lmo-m19-cnxs-001/cnxs/last.pt
DET=$(ls "$NAS_HOME"/datasets/bop_detections/default_task1/bop23_default_detections_for_task1/gdrnppdet-pbr/gdrnppdet-pbr_lmo-test_*.json)
out=$NAS_HOME/results/m19_on_dense_$V
EVAL=(.venv/bin/python src/bop_evaluate.py --dataset lmo --model_weights "$DENSE" --vit_model vit_b_16_384
      --depth_mode site --object_conditioning token --dense_heads --pose_mode pnp
      --refine_weights "$REFINER" --refine_vit_model convnext_base)
for n in 0 1 2; do
    if [[ ! -e $out/eval_valn_r$n/DONE ]]; then
        "${EVAL[@]}" --splits val_real --box_noise 0.1 --refine_iterations "$n" --out_dir "$out/eval_valn_r$n"
        .venv/bin/python scripts/val_score.py "$out/eval_valn_r$n" > "$out/eval_valn_r$n/val_score.txt"
        touch "$out/eval_valn_r$n/DONE"
    fi
    if [[ ! -e $out/eval_det_r$n/DONE ]]; then
        "${EVAL[@]}" --splits test --detections "$DET" --refine_iterations "$n" --out_dir "$out/eval_det_r$n"
        bash scripts/bop_eval.sh "$out/eval_det_r$n" fastposevitmo_lmo-test.csv > "$out/eval_det_r$n/bop_eval_test.log" 2>&1
        touch "$out/eval_det_r$n/DONE"
    fi
done
echo all done
