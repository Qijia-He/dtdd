#!/bin/bash
#SBATCH --job-name=sandbox_fusion
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/sandbox-%j.log
#SBATCH --error=logs/sandbox-%j.err
#
# Run the SandboxFusion code-execution server as its own job and record the
# address it lands on, so training jobs need not know the node in advance.
#
#   sbatch scripts/sandbox/start_server.sbatch
#
# Once the server passes its health check it writes
# "http://<node>:<port>/run_code" to $SANDBOX_ENDPOINT_FILE. Training
# launchers pick that up through scripts/sandbox/acquire_sandbox.sh.
#
# No GPU is requested on purpose: SandboxFusion is CPU-only. 16 CPUs serve
# one training job comfortably; that is also what num_workers in
# recipe/demystify/sandbox_fusion_tool_config.yaml is matched to. One
# instance per training job -- a single shared sandbox serving several
# concurrent trainings is both a single point of failure and the overload
# pattern that produces sick-but-alive 503s.
set -euo pipefail
: "${OPD_ROOT:?set OPD_ROOT to this repository}"
cd "$OPD_ROOT"
mkdir -p logs

# apptainer pull "$SIF" docker://volcengine/sandbox-fusion:server-20250609
SIF=${SANDBOX_SIF:?set SANDBOX_SIF to the sandbox-fusion image}
ENDPOINT_FILE="${SANDBOX_ENDPOINT_FILE:-$OPD_ROOT/sandbox_endpoint.txt}"
PORT=${SANDBOX_PORT:-8080}
# Hot patch: the server-20250609 image crashes when it kills a timed-out
# process (uvloop keeps writing to the dead child's stdin, and the exception
# storm wedges the event loop). Upstream fixed this in June 2025 but never
# published a newer image, so the container's own runners/base.py is
# bind-mounted with only that stdin guard applied. Build it once with:
#   apptainer exec "$SIF" cat /root/sandbox/sandbox/runners/base.py \
#     > sandbox_base_patched.py
#   patch sandbox_base_patched.py \
#     recipe_dtdd_code/overlay/sandbox_runner_stdin.patch
PATCH_FILE=${SANDBOX_RUNNER_PATCHED:-$OPD_ROOT/sandbox_base_patched.py}
PATCH_BIND="$PATCH_FILE:/root/sandbox/sandbox/runners/base.py"
[ -f "$PATCH_FILE" ] || {
  echo "ERROR: $PATCH_FILE missing; see the comment above" >&2; exit 1; }

HOST=$(hostname)
URL="http://${HOST}:${PORT}/run_code"
rm -f "$ENDPOINT_FILE"
trap 'rm -f "$ENDPOINT_FILE"' EXIT

# A real health check: HTTP 200 alone is not enough, because an orphaned
# server whose container filesystem died still answers 200 while every
# execution fails. A single sequential probe is not enough either -- one
# passed for hours while concurrent training traffic got 503s. Probe with 3
# concurrent executions and a strict timeout, requiring 2 of 3 to finish.
_healthy() {
    local tmp ok=0 i pids=()
    tmp=$(mktemp -d)
    for i in 1 2 3; do
        curl -s -m 10 "$URL" -H 'Content-Type: application/json' \
            --data-raw '{"code": "print(1)", "language": "python"}' \
            > "$tmp/$i" 2>/dev/null &
        pids+=($!)
    done
    # Wait ONLY for the probes. A bare `wait` also waits on the backgrounded
    # server, which never exits, and hangs this script forever.
    wait "${pids[@]}" 2>/dev/null
    for i in 1 2 3; do
        grep -q '"Finished"' "$tmp/$i" 2>/dev/null && ok=$((ok + 1))
    done
    rm -rf "$tmp"
    [ "$ok" -ge 2 ]
}

# Kill the server and everything it spawned. A bare `kill $SERVER_PID` only
# kills the apptainer wrapper: the containerized uvicorn survives as an
# orphan, keeps the port busy so restarts cannot bind, and its container
# filesystem is gone so all executions fail. setsid plus a group kill takes
# the whole tree down; pkill sweeps up orphans of previous instances.
_stop_server() {
    kill -- "-$SERVER_PID" 2>/dev/null || true
    sleep 3
    kill -9 -- "-$SERVER_PID" 2>/dev/null || true
    pkill -9 -u "$USER" -f 'uvicorn sandbox.server' 2>/dev/null || true
    pkill -9 -u "$USER" -f 'run-online HOST' 2>/dev/null || true
    pkill -9 -u "$USER" -f 'sandbox/scripts/run.sh' 2>/dev/null || true
    sleep 3
}

_start_server() {
    setsid apptainer run --writable-tmpfs --bind "$PATCH_BIND" "$SIF" &
    SERVER_PID=$!
    for _ in $(seq 1 90); do
        if _healthy; then
            echo "$URL" > "$ENDPOINT_FILE"
            return 0
        fi
        sleep 2
    done
    return 1
}

echo "Starting SandboxFusion at $URL ..."
SERVER_PID=0
_start_server || {
    echo "ERROR: SandboxFusion did not become healthy within the timeout" >&2
    _stop_server
    exit 1
}
echo "SandboxFusion is up: $URL  (recorded in $ENDPOINT_FILE)"

# Watchdog: health-check with real concurrent executions, and on failure kill
# the whole process tree and restart. The URL stays the same, so training
# jobs need no update. Three strikes at 20 s -> detection in about a minute.
FAIL_COUNT=0
MAX_FAILS=3
while true; do
    sleep 20
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "$(date): server exited, restarting" >&2
        rm -f "$ENDPOINT_FILE"
        _stop_server
        FAIL_COUNT=0
        _start_server || { echo "$(date): restart failed" >&2; exit 1; }
        echo "$(date): server restarted (PID $SERVER_PID)" >&2
        continue
    fi
    if _healthy; then
        FAIL_COUNT=0
        touch "$ENDPOINT_FILE"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "$(date): health check failed ($FAIL_COUNT/$MAX_FAILS)" >&2
        if [ "$FAIL_COUNT" -ge "$MAX_FAILS" ]; then
            echo "$(date): sandbox unhealthy, restarting" >&2
            rm -f "$ENDPOINT_FILE"
            _stop_server
            FAIL_COUNT=0
            _start_server || { echo "$(date): restart failed" >&2; exit 1; }
            echo "$(date): server restarted (PID $SERVER_PID)" >&2
        fi
    fi
done
