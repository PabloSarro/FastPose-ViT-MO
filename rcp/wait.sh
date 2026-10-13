#!/usr/bin/env bash
# Wait until a job is running (default) or finished, printing status changes.
# Waits at most WAIT_MINUTES (default 1, so coding assistants are not blocked): call it again to keep waiting.
#
# Usage: bash rcp/wait.sh JOB_NAME [running|done]
# Exit:  0 reached   1 job failed   2 no such job   3 still waiting   4 runai not usable   5 bad usage
if (( $# < 1 )); then
    echo "Usage: bash rcp/wait.sh JOB_NAME [running|done]" >&2
    exit 5
fi
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
name=$1
target=${2:-running}
deadline=$(( SECONDS + ${WAIT_MINUTES:-1} * 60 ))
log=$OUTPUT_DIR/runs/$name/log.txt
last=""

while true; do
    status=$(job_status "$name")
    if [[ -z "$status" ]]; then
        # A job that was just submitted can take a moment to be listed.
        if (( SECONDS < 30 )); then
            sleep 5
            continue
        fi
        echo "No job named $name (see: runai list)." >&2
        if [[ -f "$log" ]]; then
            echo "It was deleted. Its output is in $log" >&2
        fi
        exit 2
    fi
    if [[ "$status" != "$last" ]]; then
        echo "$name: $status"
        last=$status
    fi
    case "$status" in
        Failed|Error|*BackOff|*Err*)
            echo "Why: runai describe job $name" >&2
            if [[ -f "$log" ]]; then echo "Output: $log" >&2; fi
            exit 1 ;;
        Succeeded|Completed)
            if [[ -f "$log" ]]; then echo "Output: $log"; fi
            exit 0 ;;
        Running)
            if [[ "$target" == running ]]; then exit 0; fi ;;
        Terminating)
            echo "$name is being deleted." >&2
            exit 2 ;;
    esac
    if (( SECONDS >= deadline )); then
        echo "Still $status. Run this again to keep waiting." >&2
        exit 3
    fi
    sleep 10
done
