#!/bin/bash
# Babysitter for long paired runs: watch, auto-resume, evaluate, warn.
#
# Distilled from the daemon that carried this project's training campaign. The
# transferable part is not the plumbing but the failure modes it defends
# against, each of which cost real GPU-days before it was handled:
#
#   - A stalled job is invisible to the queue. When a vLLM engine child dies,
#     the parent blocks forever: SLURM keeps reporting RUNNING while the GPUs
#     idle. Liveness has to come from log mtimes, never from squeue membership.
#   - The job's .out is block-buffered and can sit untouched for hours on a
#     perfectly healthy run, so judge on the NEWEST mtime across every log.
#   - A truncated checkpoint can never load. Probe it before resuming, or the
#     retry budget burns on a checkpoint that will fail identically every time.
#   - Simultaneous resumes hammer the same filesystem during their heaviest
#     startup window. Stagger them.
#   - Quota is two independent limits. A full inode table fails writes while
#     df still shows terabytes free, so check both.
#   - Evaluate at save time. Every coverage hole in our record came from
#     "train now, evaluate later"; see tools/autoeval.py.
#
# Runs as a batch job so it survives an interactive session ending. Configure
# with the environment variables below and the ARMS table; the quota probe is
# the one thing you will likely need to adapt to your site.
#
#   OPD_ROOT          this repo
#   MONITOR_ACCOUNT   sbatch account for auto-resubmissions
#   MONITOR_PARTITION default partition for them
#   POLL_SEC          seconds between passes (default 600)
#   STALL_MIN         all-logs-quiet minutes before declaring a zombie (default 60)
#   MAXRESUME         per-arm resume cap (default 3)
#   NORESUME=1        watch and report only, never resubmit
set -uo pipefail

OPD=${OPD_ROOT:?set OPD_ROOT to the repo checkout}
MONITOR_ACCOUNT=${MONITOR_ACCOUNT:?set an sbatch account}
MONITOR_PARTITION=${MONITOR_PARTITION:?set an sbatch partition}
POLL_SEC=${POLL_SEC:-600}
STALL_MIN=${STALL_MIN:-60}   # healthy gaps of ~25 min are normal; in-run eval adds more
MAXRESUME=${MAXRESUME:-3}    # so a hard-broken arm cannot loop forever
NORESUME=${NORESUME:-0}
LOG=$OPD/monitor.log
STATUS=$OPD/STATUS.txt

# arm name -> "checkpoint dir|submit script|target step"
# The submit script is re-run to resume; every setting in scripts/train/settings/
# keeps all checkpoints, so resuming is safe.
declare -A ARMS=(
  [dtdd_15k]="$OPD/checkpoint/dtddB_15k|$OPD/scripts/train/settings/submit_18k.sh|279"
  [opd_15k]="$OPD/checkpoint/opdB_15k|$OPD/scripts/train/settings/submit_18k.sh|279"
)
declare -A NRESUME

say(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }

# Newest training step this arm reached, from its run logs.
arm_step(){
  grep -aoE 'training/global_step:[0-9]+' "$OPD"/logs/*"$1"*.out 2>/dev/null \
    | grep -oE '[0-9]+$' | tail -1
}

# Newest mtime across everything this arm writes -- the liveness signal.
arm_age_min(){
  local newest=0 m
  for f in "$OPD"/logs/*"$1"*.out "$OPD"/logs/*"$1"*.err; do
    [ -f "$f" ] || continue
    m=$(stat -c %Y "$f" 2>/dev/null || echo 0)
    [ "$m" -gt "$newest" ] && newest=$m
  done
  [ "$newest" = 0 ] && { echo -1; return; }
  echo $(( ($(date +%s) - newest) / 60 ))
}

# Newest checkpoint that actually loads. A quota-truncated save leaves a shard
# with no zip central directory; resuming from it dies in minutes, every time.
newest_loadable_step(){
  python3 - "$1" <<'PY'
import glob, re, sys, zipfile
ds = sorted(glob.glob(sys.argv[1] + "/global_step_[0-9]*"),
            key=lambda d: int(re.search(r"(\d+)$", d).group(1)), reverse=True)
for d in ds:
    shards = glob.glob(d + "/actor/*_world_size_*.pt")
    try:
        ok = bool(shards) and all(zipfile.ZipFile(f).namelist() for f in shards)
    except Exception:
        ok = False
    if ok:
        print(re.search(r"(\d+)$", d).group(1)); break
PY
}

# Both quota limits. Adapt the inode probe to your filesystem's tooling --
# df -i is the portable fallback and is what this uses.
quota_sentinel(){
  local free_g free_i
  free_g=$(df -BG --output=avail "$OPD" 2>/dev/null | tail -1 | tr -dc '0-9')
  free_i=$(df -i --output=iavail "$OPD" 2>/dev/null | tail -1 | tr -dc '0-9')
  [ -n "$free_g" ] && [ "$free_g" -lt 500 ] && \
    say "DISK WARN: ${free_g}G free (<500G) -- checkpoint saves will start failing"
  [ -n "$free_i" ] && [ "$free_i" -lt 200000 ] && \
    say "INODE WARN: ${free_i} inodes left (<200k) -- writes fail randomly while df still looks fine"
}

# Only one monitor may run: two of them will each resubmit the same dead arm.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  other=$(squeue -u "$USER" -h -o "%i %j %T" 2>/dev/null \
          | awk -v me="$SLURM_JOB_ID" '$2=="opd_monitor" && $3=="RUNNING" && $1+0 < me+0 {print $1}' | head -1)
  if [ -n "$other" ]; then
    say "monitor $SLURM_JOB_ID exiting: older monitor $other already running"
    exit 0
  fi
fi

# On a preemptible or short-walltime queue one monitor job cannot cover a
# multi-day run. Queue a successor that starts only once this one exits
# (afterany, so preemption triggers it too); the singleton guard above keeps
# them from overlapping.
RENEWED=0
renew_if_expiring(){
  [ -n "${SLURM_JOB_ID:-}" ] || return 0
  [ "$RENEWED" = 1 ] && return 0
  [ -n "${MONITOR_SELF_SBATCH:-}" ] || return 0
  local left secs
  left=$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null | tr -d ' ')
  [ -n "$left" ] || return 0
  secs=$(python3 -c "
import sys
t=sys.argv[1]; d=0
if '-' in t: d,t=t.split('-',1)
p=[int(x) for x in t.split(':')]
while len(p)<3: p.insert(0,0)
print(int(d)*86400+p[0]*3600+p[1]*60+p[2])" "$left" 2>/dev/null) || return 0
  if [ "$secs" -lt 1200 ]; then
    local new
    new=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" "$MONITOR_SELF_SBATCH" 2>/dev/null)
    [ -n "$new" ] && { say "self-renew: ${secs}s left -> successor $new"; RENEWED=1; }
  fi
}

say "monitor started (pid $$, job ${SLURM_JOB_ID:-none})"
while true; do
  renew_if_expiring

  { echo "=== STATUS $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo; echo "--- QUEUE ---"
    squeue -u "$USER" -o "%.10i %.20j %.9T %.10M %.13R" 2>/dev/null
    echo; echo "--- TRAINING ---"
    for a in "${!ARMS[@]}"; do
      IFS='|' read -r d sub tgt <<< "${ARMS[$a]}"
      printf "  %-16s step %s/%s\n" "$a" "$(arm_step "$a")" "$tgt"
    done
    echo; echo "--- HELD-OUT RESULTS ---"
    python3 - <<'PY' 2>/dev/null
import json, os
p = os.environ["OPD_ROOT"] + "/results/eval_results.jsonl"
if os.path.exists(p):
    d = {}
    for line in open(p):
        r = json.loads(line)
        if r.get("metric") == "mean_score":
            d.setdefault((r["arm"], r["step"]), {})[r["benchmark"]] = r["value"]
    for (arm, st), v in sorted(d.items()):
        print(f"  {arm:>12} s{st:<5} " + "  ".join(f"{k}={x:.3f}" for k, x in sorted(v.items())))
PY
  } > "$STATUS.tmp" && mv "$STATUS.tmp" "$STATUS"

  # ---- zombie detection ----
  for a in "${!ARMS[@]}"; do
    jid=$(squeue -u "$USER" -h -o "%i %j %T" 2>/dev/null | awk -v n="$a" '$2==n && $3=="RUNNING"{print $1}' | head -1)
    [ -z "$jid" ] && continue
    age=$(arm_age_min "$a"); [ "$age" -lt 0 ] && continue
    # A fatal signal in stderr means the engine child is already dead, so there
    # is nothing to wait for -- 10 quiet minutes is confirmation enough.
    if grep -aqs "Fatal Python error" "$OPD"/logs/*"$a"*.err 2>/dev/null && [ "$age" -ge 10 ]; then
      say "FASTKILL $a (job $jid): fatal signal in stderr, logs quiet ${age}m -> scancel"
      [ "$NORESUME" = 1 ] || scancel "$jid" 2>/dev/null
      continue
    fi
    if [ "$age" -ge "$STALL_MIN" ]; then
      say "STALL $a (job $jid): all logs quiet ${age}m -> scancel; next pass resumes"
      [ "$NORESUME" = 1 ] || scancel "$jid" 2>/dev/null
    fi
  done

  # ---- crash detection + staggered auto-resume ----
  NSUB=0
  for a in "${!ARMS[@]}"; do
    [ "$NORESUME" = 1 ] && break
    IFS='|' read -r d sub tgt <<< "${ARMS[$a]}"
    # Exact job-name match: a prefix match lets one arm of a pair mask its
    # partner, so the dead one is never noticed.
    [ "$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -cx "$a")" -gt 0 ] && continue
    step=$(arm_step "$a")
    [ -z "$step" ] && continue          # never started -> leave it alone
    [ "$step" -ge "$tgt" ] && continue  # finished
    n=${NRESUME[$a]:-0}
    if [ "$n" -ge "$MAXRESUME" ]; then
      say "$a stopped at step $step, resume cap ($MAXRESUME) reached -- needs a human"; continue
    fi
    last=$(newest_loadable_step "$d")
    if [ -z "$last" ]; then
      say "$a stopped at step $step but no loadable checkpoint found -- needs a human"; continue
    fi
    # Stagger: concurrent restarts collide during their heaviest startup I/O.
    jid=$(sbatch --parsable --job-name="$a" --account="$MONITOR_ACCOUNT" \
            --partition="$MONITOR_PARTITION" --begin=now+$((NSUB*6))minutes \
            --export=ALL,RESUME_STEP="$last" "$sub" 2>/dev/null)
    NRESUME[$a]=$((n+1)); NSUB=$((NSUB+1))
    say "AUTO-RESUME $a from step $last -> job ${jid:-FAILED} (attempt $((n+1))/$MAXRESUME, +$(( (NSUB-1)*6 ))m)"
  done

  # ---- checkpoint saved => eval submitted => results harvested ----
  quota_sentinel
  ae=$(python3 "$OPD/tools/autoeval.py" --apply 2>&1)
  echo "$ae" | grep -q SUBMITTED && say "AUTOEVAL: $(echo "$ae" | grep -c SUBMITTED) new eval(s) submitted"
  python3 "$OPD/tools/harvest.py" >/dev/null 2>&1

  sleep "$POLL_SEC"
done
