# Shared by the scripts in this folder. You should not need to edit this file.
#
# Exit codes used by all rcp/ scripts:
#   0 ok   1 job failed   2 no such job   3 still waiting   4 runai not usable   5 invalid request
set -euo pipefail
export SUPPRESS_DEPRECATION_MESSAGE=true   # hides runai's "CLI v1 is now deprecated" notice
RCP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

GASPAR_USERNAME=${GASPAR_USERNAME:-$(id -un)}
NAS_HOME=${NAS_HOME:-/mnt/cvlab/scratch/cvlab/home/$GASPAR_USERNAME}
JOB_HOME=$NAS_HOME/.home
if [[ -n "${RCP_JOB:-}" ]]; then
    PATH=$JOB_HOME/.local/bin:$PATH
fi

# The project is the folder that contains rcp/ (or the folder given by `dev` and `train`).
# Its name is used for the output folder. In a Git worktree, it is the main repository's name.
PROJECT_ROOT=${RCP_WORK_DIR:-$(dirname "$RCP_DIR")}
PROJECT_NAME=$(basename "$PROJECT_ROOT")
if git_common=$(git -C "$PROJECT_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null); then
    PROJECT_NAME=$(basename "$(dirname "$git_common")")
fi
WORK_DIR=$PROJECT_ROOT

# Settings from project.env. Folders not set there get the defaults below.
unset DATA_DIR OUTPUT_DIR
source "$RCP_DIR/project.env"
DATA_DIR=${DATA_DIR:-$NAS_HOME/datasets}
OUTPUT_DIR=${OUTPUT_DIR:-$NAS_HOME/outputs/$PROJECT_NAME}

if [[ "$WORK_DIR" != /* ]]; then
    echo "WORK_DIR must be an absolute path: $WORK_DIR" >&2
    exit 5
fi
mkdir -p "$OUTPUT_DIR/runs" "$JOB_HOME/.local/bin"
if ! mkdir -p "$DATA_DIR" 2>/dev/null; then
    echo "Warning: DATA_DIR does not exist and could not be created: $DATA_DIR" >&2
fi

# On the jumphost, keep a copy of runai on the NAS for jobs. Only the size and
# date are compared, so this costs nothing unless RCP updated runai.
if [[ -z "${RCP_JOB:-}" ]]; then
    runai_bin=""
    for candidate in $(type -ap runai); do
        if [[ "$(readlink -f "$candidate")" != "$(readlink -f "$JOB_HOME/.local/bin/runai" 2>/dev/null)" ]]; then
            runai_bin=$candidate
            break
        fi
    done
    if [[ -z "$runai_bin" ]]; then
        echo "runai not found. Run this on the jumphost, or inside a job started with dev." >&2
        exit 4
    fi
    if [[ "$(stat -c %s.%Y "$runai_bin")" != "$(stat -c %s.%Y "$JOB_HOME/.local/bin/runai" 2>/dev/null)" ]]; then
        cp -p "$runai_bin" "$JOB_HOME/.local/bin/runai"
    fi
fi

# Check that runai works, and say clearly what to do when it doesn't.
if ! runai_error=$(runai list 2>&1 >/dev/null); then
    echo "runai does not work: $(head -c 300 <<<"${runai_error:-unknown error}")" >&2
    if grep -qiE 'login|token|authenticat|unauthori|expired' <<<"$runai_error"; then
        echo "The Run:AI login has expired. On the jumphost, run: runai login" >&2
    elif [[ -n "${RCP_JOB:-}" ]]; then
        printf 'Run:AI cannot be reached from this job. Run this on the jumphost instead:\n  cd %q && GPUS=%q CPUS=%q MEMORY=%q %sbash %q' \
            "$PROJECT_ROOT" "$GPUS" "$CPUS" "$MEMORY" "${SNAPSHOT:+SNAPSHOT=$SNAPSHOT }" "$RCP_DIR/$(basename "$0")" >&2
        printf ' %q' "$@" >&2
        printf '\n' >&2
    fi
    exit 4
fi

job_exists() {
    runai describe job "$1" >/dev/null 2>&1
}

job_status() {
    runai describe job "$1" 2>/dev/null | awk '/^Status:/ { print $2; exit }' || true
}

# Job names: lowercase letters, digits and '-', starting and ending with a letter or digit.
check_name() {
    if [[ ! "$1" =~ ^[a-z0-9]([-a-z0-9]{0,38}[a-z0-9])?$ ]]; then
        echo "Invalid job name '$1': use lowercase letters, digits and '-', at most 40 characters (e.g. lr-001)." >&2
        exit 5
    fi
}

# One line per job started or stopped, in $OUTPUT_DIR/jobs.log.
log_event() {
    printf '%s  %-6s %-30s by %s%s  %s\n' "$(date '+%F %T')" "$1" "$2" "$GASPAR_USERNAME" \
        "${RCP_JOB:+ from job $RCP_JOB}" "${3:-}" >> "$OUTPUT_DIR/jobs.log"
}

# Usage: submit_job NAME START_DIR [MORE RUNAI OPTIONS] --command -- COMMAND...
# The NAS is mounted at the same path inside the job as on the jumphost.
# HOME is $NAS_HOME/.home, so logins, tools (uv, Claude, Codex, runai) and caches persist.
# Build and compile caches (cargo, ccache, uv, Triton, torch.compile) live in
# $NAS_HOME/.cache. They are set here because --command bypasses the image entrypoint.
submit_job() {
    local name=$1 dir=$2
    shift 2
    local login=()
    if [[ -n "${KUBECONFIG:-}" ]]; then
        login=(--environment "KUBECONFIG=$KUBECONFIG")
    fi
    runai submit --name "$name" \
        --image "$IMAGE" \
        --gpu "$GPUS" \
        --cpu "$CPUS" --cpu-limit "$CPUS" \
        --memory "$MEMORY" --memory-limit "$MEMORY" \
        --node-pool "$NODE_POOL" \
        --run-as-uid "$(id -u)" --run-as-gid "$(id -g)" \
        --existing-pvc claimname=cvlab-scratch,path=/mnt/cvlab/scratch \
        --working-dir "$dir" \
        --large-shm \
        --environment "RCP_JOB=$name" \
        --environment "GASPAR_USERNAME=$GASPAR_USERNAME" \
        --environment "NAS_HOME=$NAS_HOME" \
        --environment "HOME=$JOB_HOME" \
        --environment "DATA_DIR=$DATA_DIR" \
        --environment "OUTPUT_DIR=$OUTPUT_DIR" \
        --environment "PYTHONUNBUFFERED=1" \
        --environment "PYTHONNOUSERSITE=1" \
        --environment "CARGO_HOME=$NAS_HOME/.cache/cargo" \
        --environment "CCACHE_DIR=$NAS_HOME/.cache/ccache" \
        --environment "UV_CACHE_DIR=$NAS_HOME/.cache/uv" \
        --environment "TRITON_CACHE_DIR=$NAS_HOME/.cache/triton" \
        --environment "TORCHINDUCTOR_CACHE_DIR=$NAS_HOME/.cache/torchinductor" \
        "${login[@]}" \
        "$@"
}
