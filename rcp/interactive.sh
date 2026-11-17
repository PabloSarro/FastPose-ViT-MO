#!/usr/bin/env bash
# Start a job for interactive work and open a shell in it. Run it on the jumphost.
# Run it again to get another shell in the same job.
#
# Usage: bash rcp/interactive.sh [JOB_NAME]      (default: DEV_JOB in project.env, else dev)
# Stop:  runai delete job JOB_NAME
if [[ -n "${RCP_JOB:-}" ]]; then
    echo "You are already inside the job $RCP_JOB. Run this on the jumphost." >&2
    exit 5
fi
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
name=${1:-${DEV_JOB:-dev}}
check_name "$name"

# A job with this name that is still being deleted must be gone before we start a new one.
if [[ "$(job_status "$name")" == Terminating ]]; then
    echo "The previous $name job is still stopping. Waiting until it is gone..."
    while job_exists "$name"; do sleep 5; done
fi
# Interactive jobs use the live code, so edits are visible right away.
if ! job_exists "$name"; then
    submit_job "$name" "$WORK_DIR" --interactive --command -- sleep infinity
    echo "Waiting for $name to start. It waits for a free GPU, and the first job on a"
    echo "machine also downloads the image, which can take several minutes."
fi
WAIT_MINUTES=30 bash "$RCP_DIR/wait.sh" "$name" running
# The `dev` command asks for the shell to open in the folder it was typed in.
# The job's shell startup (env.sh, from setup-jumphost.sh) reads this note once.
# Not when project.env points WORK_DIR somewhere else, e.g. code packaged in the image.
if [[ -n "${RCP_SHELL_DIR:-}" && "$WORK_DIR" == "$PROJECT_ROOT" ]]; then
    printf '%s\n' "$RCP_SHELL_DIR" > "$JOB_HOME/.rcp-cd"
fi
exec runai bash "$name"
