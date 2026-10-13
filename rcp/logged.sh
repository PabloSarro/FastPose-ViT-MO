#!/usr/bin/env bash
# Runs inside a training job (started by train.sh). Saves everything the command
# prints to $RUN_DIR/log.txt, so it is kept after the job is deleted.
# If the cluster restarts the job, the new attempt is appended to the same log.
set -o pipefail
log=$RUN_DIR/log.txt
echo "=== $(date '+%F %T') start: $*" >> "$log"
"$@" 2>&1 | tee -a "$log"
status=${PIPESTATUS[0]}
echo "=== $(date '+%F %T') exit code $status" >> "$log"
exit "$status"
