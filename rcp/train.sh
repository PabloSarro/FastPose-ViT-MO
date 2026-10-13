#!/usr/bin/env bash
# Start a training job: COMMAND runs on its own GPU and keeps going after you log out.
#
# Usage:   bash rcp/train.sh JOB_NAME COMMAND [ARGS...]
# Example: bash rcp/train.sh lr-001 .venv/bin/python train.py --lr 0.001
#
# Each run gets its own folder, $OUTPUT_DIR/runs/JOB_NAME, that the job sees as $RUN_DIR:
#   code/      copy of the code at submission (files Git would commit; .venv is linked)
#   log.txt    everything COMMAND prints, kept after the job is deleted
#   info.txt   what was run, when, with which settings and Git commit
# Save results and checkpoints in $RUN_DIR. SNAPSHOT=0 runs the live code instead of a copy.
# Names are never reused, so earlier runs are never overwritten.
if (( $# < 2 )); then
    echo "Usage: bash rcp/train.sh JOB_NAME COMMAND [ARGS...]" >&2
    exit 5
fi
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
name=$1
shift
check_name "$name"

if [[ "$name" == "${RCP_JOB:-}" ]]; then
    echo "'$name' is the job you are running in. Pick another name." >&2
    exit 5
fi
run_dir=$OUTPUT_DIR/runs/$name
# mkdir without -p fails if the folder exists, so two runs can never claim the same name.
if job_exists "$name" || ! mkdir "$run_dir" 2>/dev/null; then
    echo "The name '$name' is already used ($run_dir or a running job). Pick a new name." >&2
    exit 5
fi

start_dir=$WORK_DIR
extra=(--environment "RUN_DIR=$run_dir")
if [[ "${SNAPSHOT:-1}" == 1 ]] && git -C "$WORK_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    start_dir=$run_dir/code
    mkdir -p "$start_dir"
    (cd "$WORK_DIR" && git ls-files -z --cached --others --exclude-standard -- . ':(exclude).claude/worktrees' \
        | tar --null --ignore-failed-read --no-recursion -T - -cf -) | tar -xf - -C "$start_dir"
    if [[ -e "$WORK_DIR/.venv" ]]; then
        ln -s "$WORK_DIR/.venv" "$start_dir/.venv"
    fi
    # Imports come from the copy, even if the project is installed in editable mode.
    extra+=(--environment "PYTHONPATH=$start_dir/src:$start_dir")
elif [[ ! -d "$WORK_DIR" ]]; then
    echo "The job runs the code inside the image, in $WORK_DIR."
elif [[ "${SNAPSHOT:-1}" == 1 ]]; then
    echo "Not a Git repository, so the job runs the live code in $WORK_DIR."
fi

{
    echo "job:      $name"
    echo "started:  $(date '+%F %T') by $GASPAR_USERNAME${RCP_JOB:+ from job $RCP_JOB}"
    echo "command:  $*"
    echo "source:   $WORK_DIR"
    echo "folder:   $start_dir"
    echo "image:    $IMAGE"
    echo "size:     $GPUS GPU, $CPUS CPU, $MEMORY RAM, node pool $NODE_POOL"
    echo "git:      $(git -C "$WORK_DIR" rev-parse HEAD 2>/dev/null || echo none)"
    git -C "$WORK_DIR" status --short 2>/dev/null | sed 's/^/  changed: /'
} > "$run_dir/info.txt"

if ! submit_error=$(submit_job "$name" "$start_dir" "${extra[@]}" --command -- bash "$RCP_DIR/logged.sh" "$@" 2>&1); then
    echo "runai could not start the job: $submit_error" >&2
    # Remove only the folder this run just created, so the name can be used again.
    rm -rf -- "$run_dir"
    exit 4
fi
log_event start "$name" "$*"
cat <<EOF
Run folder: $run_dir
Status:     bash rcp/wait.sh $name done
Output:     tail -n 50 $run_dir/log.txt     (startup errors: runai logs $name)
Stop:       bash rcp/stop.sh $name
EOF
