#!/bin/bash
# Source this from a training or evaluation job to give it a code-execution
# sandbox:
#
#   source scripts/sandbox/acquire_sandbox.sh
#
# Order of preference:
#   1. SANDBOX_FUSION_URL already set and healthy (a shared or cloud
#      instance) -- used as is, no job submitted.
#   2. A dedicated scripts/sandbox/start_server.sbatch job for this job,
#      excluding nodes already running your jobs, with a matching time limit.
#      Cancelled on exit.
#   3. Otherwise the job fails early, which beats training against a dead
#      sandbox.
#
# This registers `scancel` on EXIT. If the caller sets its own EXIT trap
# after sourcing this, it REPLACES this one -- include
# `scancel $SANDBOX_JOB_ID 2>/dev/null || true` in that trap.

_sbx_healthy() {
    local response
    local probe='{"code": "print(1)", "language": "python"}'
    response=$(curl -s -m 10 "$1" -H 'Content-Type: application/json' \
        --data-raw "$probe" 2>/dev/null || true)
    [[ "$response" == *'"Finished"'* ]]
}

if [ -n "${SANDBOX_FUSION_URL:-}" ]; then
    if _sbx_healthy "$SANDBOX_FUSION_URL"; then
        echo "acquire_sandbox: using preset endpoint $SANDBOX_FUSION_URL"
        export SANDBOX_JOB_ID=""
        return 0 2>/dev/null || exit 0
    fi
    echo "acquire_sandbox: WARNING preset SANDBOX_FUSION_URL failed its" \
         "health check; starting a dedicated instance instead" >&2
    unset SANDBOX_FUSION_URL
fi

SANDBOX_PARTITION="${SANDBOX_PARTITION:-}"
SANDBOX_WAIT_MINUTES="${SANDBOX_WAIT_MINUTES:-20}"
SANDBOX_ENDPOINT_FILE="$OPD_ROOT/sandbox_endpoint_job${SLURM_JOB_ID}.txt"
export SANDBOX_ENDPOINT_FILE
rm -f "$SANDBOX_ENDPOINT_FILE"

# Keep the sandbox off nodes already running your jobs: a sandbox sharing a
# node with a training job is the other half of the overload pattern.
_sbx_exclude=$(squeue -u "$USER" -h -t RUNNING -o "%N" | sort -u | paste -sd, -)
# Match the sandbox's lifetime to this job's remaining time, so a
# maintenance window blocks both or neither.
_sbx_time=$(squeue -j "$SLURM_JOB_ID" -h -o "%l" | head -1)

SANDBOX_JOB_ID=$(sbatch --parsable \
    --job-name="sandbox_${SLURM_JOB_ID}" \
    ${SANDBOX_PARTITION:+--partition="$SANDBOX_PARTITION"} \
    ${_sbx_time:+--time="$_sbx_time"} \
    ${_sbx_exclude:+--exclude="$_sbx_exclude"} \
    --export=ALL,SANDBOX_ENDPOINT_FILE="$SANDBOX_ENDPOINT_FILE" \
    "$OPD_ROOT/scripts/sandbox/start_server.sbatch")
export SANDBOX_JOB_ID
echo "acquire_sandbox: submitted sandbox job $SANDBOX_JOB_ID" \
     "(exclude=${_sbx_exclude:-none}, time=${_sbx_time:-default})"
trap 'scancel "$SANDBOX_JOB_ID" 2>/dev/null || true' EXIT

_sbx_deadline=$((SECONDS + SANDBOX_WAIT_MINUTES * 60))
while [ ! -s "$SANDBOX_ENDPOINT_FILE" ] &&
      [ "$SECONDS" -lt "$_sbx_deadline" ]; do
    sleep 10
done

if [ -s "$SANDBOX_ENDPOINT_FILE" ]; then
    SANDBOX_FUSION_URL="$(cat "$SANDBOX_ENDPOINT_FILE")"
    export SANDBOX_FUSION_URL
    if _sbx_healthy "$SANDBOX_FUSION_URL"; then
        echo "acquire_sandbox: ready at $SANDBOX_FUSION_URL" \
             "(job $SANDBOX_JOB_ID)"
        return 0 2>/dev/null || exit 0
    fi
    echo "acquire_sandbox: WARNING $SANDBOX_FUSION_URL failed its health" \
         "check" >&2
fi

echo "ERROR: acquire_sandbox: no healthy sandbox after" \
     "${SANDBOX_WAIT_MINUTES}m (job $SANDBOX_JOB_ID). Aborting early." >&2
scancel "$SANDBOX_JOB_ID" 2>/dev/null || true
return 1 2>/dev/null || exit 1
