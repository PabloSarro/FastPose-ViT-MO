#!/usr/bin/env bash
# Delete a training job of this project (started by train.sh) and record it in jobs.log.
# Its run folder (code copy, log, results) is kept.
#
# Usage: bash rcp/stop.sh JOB_NAME
#
# It refuses the job you are running in, and jobs that are not training runs of this
# project. To delete your interactive job, use: runai delete job dev (on the jumphost).
if (( $# != 1 )); then
    echo "Usage: bash rcp/stop.sh JOB_NAME" >&2
    exit 5
fi
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
name=$1
check_name "$name"

if [[ "$name" == "${RCP_JOB:-}" ]]; then
    echo "'$name' is the job you are running in. It can't be stopped from inside. The user deletes it." >&2
    exit 5
fi
run_dir=$OUTPUT_DIR/runs/$name
if [[ ! -d "$run_dir" ]]; then
    echo "'$name' is not a training job of this project (no $run_dir). Not stopped." >&2
    exit 5
fi
if ! job_exists "$name"; then
    echo "'$name' is already gone. Its output is in $run_dir/log.txt."
    exit 0
fi
# A job with the same name could belong to another project: check where it writes.
command_line=$(runai describe job "$name" 2>/dev/null | grep -m1 '^Command Line:' || true)
if [[ " $command_line " != *" RUN_DIR=$run_dir "* ]]; then
    echo "The running job '$name' does not use $run_dir: it belongs to another project or worktree. Not stopped." >&2
    exit 5
fi

runai delete job "$name"
log_event stop "$name"
echo "Stopped $name. Its output is kept in $run_dir/log.txt."
