#!/usr/bin/env bash
# Inside a train job: continue the run of an earlier, stopped job under this job's new name.
# Job names are never reused, so `--resume auto` alone would find no checkpoint and start over.
# This reads the earlier job's command from its info.txt, copies its variant folder (last.pt,
# model.pth, logs) into this job's RUN_DIR once, then runs the same command. The earlier run
# folder is only read. Works for the one-variant scripts (m18_dense.sh, m10_ablations.sh), whose
# last argument is the variant and its folder name. A resumed job can be resumed again.
#
# Usage (from the project root), with the size of the earlier job (docs/PAUSED_RUNS.md):
#   GPUS=1 CPUS=16 MEMORY=32Gi bash rcp/train.sh lmo-m18-densex5m05-002 \
#       bash scripts/resume_run.sh lmo-m18-densex5m05-001
# The new job runs the current code: check that training did not change since the earlier job.
set -euo pipefail
: "${RUN_DIR:?run inside a train job (RUN_DIR not set)}"
(( $# == 1 )) || { echo "usage: $0 EARLIER_JOB"; exit 5; }
runs=$(dirname "$RUN_DIR")

# Follow a chain of resumed jobs back to the original command, and take the checkpoint of the
# latest job of the chain that saved one.
job=$1 src=
while :; do
    info=$runs/$job/info.txt
    [[ -f $info ]] || { echo "no $info"; exit 5; }
    command=$(sed -n 's/^command:  //p' "$info")
    if [[ $command == "bash scripts/resume_run.sh "* ]]; then
        variant_folders=("$runs/$job"/*/last.pt)
        [[ -z $src && -e ${variant_folders[0]} ]] && src=$job
        job=${command##* }
        continue
    fi
    [[ $command == "bash -c "* ]] || { echo "unexpected command in $info: $command"; exit 5; }
    command=${command#bash -c }
    break
done
variant=${command##* }
[[ -n $src ]] || src=$job
old=$runs/$src/$variant

if [[ -e $RUN_DIR/$variant ]]; then  # this job restarted: its own checkpoint is the newest
    echo "== resume: $RUN_DIR/$variant exists, keeping it"
elif [[ -e $old/last.pt ]]; then
    echo "== resume: copying $old ($(date '+%F %T'))"
    # Copy aside, then rename: a job killed during the copy does not leave a partial checkpoint
    mkdir -p "$RUN_DIR/$variant.part"
    cp -a "$old/." "$RUN_DIR/$variant.part/"
    mv "$RUN_DIR/$variant.part" "$RUN_DIR/$variant"
else
    echo "== resume: $old has no checkpoint, starting from scratch"
fi
echo "== resume: running: $command"
exec bash -c "$command"
