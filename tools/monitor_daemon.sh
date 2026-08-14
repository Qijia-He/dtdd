#!/bin/bash
# ----------------------------------------------------------------------------
# SITE-SPECIFIC REFERENCE, NOT PART OF THE REPRODUCTION PATH.
#
# This is the live monitor daemon from our 2026-07/08 training campaign on a
# SLURM + GPFS cluster (UW Hyak). It watched BOTH lanes of the project at
# once: the math lane in this repo and an agentic lane (the tcod_* arms and
# the $AG/$CSE paths below belong to that separate codebase). It is shipped
# near-verbatim because the patterns are the transferable part:
#
#   - stall detection by newest-log-mtime, not queue state (a dead vLLM child
#     process leaves the SLURM job RUNNING while its GPUs idle indefinitely)
#   - FASTKILL on fatal signals in stderr (no point waiting the full window)
#   - singleton guard + self-renew for preemptible / short-walltime queues
#   - staggered resumes (simultaneous restarts hammered GPFS into SIGBUS)
#   - probing that a checkpoint actually LOADS (zip central directory check)
#     before resuming from it, so a truncated save can't eat the retry budget
#   - checkpoint saved => eval submitted (tools/autoeval.py) => results
#     harvested (tools/harvest.py), so results survive checkpoint pruning
#   - disk AND inode sentinels (a file-count quota kills saves while df still
#     shows terabytes free)
#
# Adapt paths, account, partitions, and the ARMS table before running any of
# this. hyakalloc / hyakstorage are UW Hyak commands.
# ----------------------------------------------------------------------------
# Persistent experiment monitor + auto-resume daemon.
# Runs as a SLURM job so it survives interactive-session teardown.
#   - polls every 10 min
#   - detects: job died / training stalled
#   - AUTO-RESUMES crashed training arms (safe: all configs keep every ckpt)
#   - appends to $OPD/monitor.log and refreshes $OPD/STATUS.txt
OPD=${OPD_ROOT:?set OPD_ROOT to the repo checkout}
CSE=${SCRATCH2:?second scratch root (agentic-lane harness + logs)}
AG=${AGENT_CKPTS:?agentic-lane checkpoint root}
MONITOR_ACCOUNT=${MONITOR_ACCOUNT:?sbatch account for auto-resumes}
LOG=$OPD/monitor.log
STATUS=$OPD/STATUS.txt
MAXRESUME=3                       # per-arm cap, so a hard-broken arm can't loop forever
# Auto-resume is ON, but resume is FLAKY, in two distinct ways -- hence STALL_MIN below.
#  (a) starvation: if the explorer's first sync goes through _pull_latest_weights
#      instead of _checkpoint_weights_update, the vLLM rollout model keeps
#      model_version=0 (vllm_model.py:141 "TODO: resume the value from the
#      checkpoint"). The trainer demands min_model_version = step - max_staleness, so
#      QueueBuffer.get() silently drops every experience and the trainer dies on an
#      empty batch. Look for "Redirect `lora_path` from old_model_version=0 to N" --
#      present = healthy, absent (only "No new model weights found") = doomed.
#  (b) SIGBUS hang: the vLLM engine core is a spawn()ed child, NOT a Ray actor. If it
#      dies (e.g. mmap of the LoRA adapter faulting when the fs is out of quota), the
#      parent blocks forever on add_lora, SLURM keeps reporting RUNNING, and 4 GPUs
#      idle indefinitely. Only a liveness check catches this -- a queue-membership
#      check never will.
NORESUME=0
STALL_MIN=60   # all logs quiet N min -> declare zombie (healthy jobs measured up to
               # 24 min of silence; an in-run eval adds ~35 -- hence 60)

# arm  ->  "ckpt_dir|config|partition|gpus"   (agentic lane; separate codebase)
declare -A ARMS=(
 [opd_v2]="$AG/ALFWORLD_TCOD/alfworld_opd_lora1e4_v2|TCOD_examples/alfworld/opd_lora_lr1e4_v2_resume_ext400.yaml|gpu-a40|4"
 [dtdd_v2]="$AG/ALFWORLD_TCOD/alfworld_dtdd_lora1e4_v2|TCOD_examples/alfworld/dtdd_lora_lr1e4_v2_ext400.yaml|gpu-a40|4"
 [fullft_opd]="$AG/ALFWORLD_TCOD/alfworld_opd_fullft|TCOD_examples/alfworld/alfworld_opd_fullft_ext400.yaml|gpu-a40|4"
 [fullft_dtdd]="$AG/ALFWORLD_TCOD/alfworld_dtdd_fullft|TCOD_examples/alfworld/alfworld_dtdd_fullft_ext400.yaml|gpu-a40|4"
 [ws120k_opd]="$AG/WEBSHOP_TCOD/webshop_opd_120k|TCOD_examples/webshop/opd_lora.yaml|gpu-l40s|4"
 [ws120k_dtdd]="$AG/WEBSHOP_TCOD/webshop_dtdd_120k|TCOD_examples/webshop/dtdd_lora.yaml|gpu-l40s|4"
 [gap14x_dtdd]="$AG/ALFWORLD_TCOD/alfworld_dtdd_gap14x|TCOD_examples/alfworld/alfworld_dtdd_gap14x.yaml|gpu-a40|4"
)
declare -A NRESUME
say(){ echo "[$(date '+%m-%d %H:%M')] $*" >> $LOG; }

# ---- disk sentinel: the lab share once hit 100% and killed both arms of a pair
# mid-save. Warn loudly at 97% space or files so the next fill-up is caught while
# checkpoints still write. (hyakstorage is UW Hyak; substitute your quota tool.)
disk_sentinel(){
  local line=$(hyakstorage $(dirname $OPD) 2>/dev/null | grep "Total:")
  local used=$(echo "$line" | grep -oE "[0-9]+GB / [0-9]+GB" | head -1)
  local u=$(echo "$used" | grep -oE "^[0-9]+"); local t=$(echo "$used" | grep -oE "/ [0-9]+" | grep -oE "[0-9]+")
  local files=$(echo "$line" | grep -oE "[0-9]+ / [0-9]+ files" | head -1)
  local fu=$(echo "$files" | grep -oE "^[0-9]+"); local ft=$(echo "$files" | grep -oE "/ [0-9]+" | grep -oE "[0-9]+")
  if [ -n "$u" ] && [ -n "$t" ] && [ $((u * 100)) -ge $((t * 97)) ]; then
    say "DISK ALERT: space ${u}GB/${t}GB (>=97%) — checkpoint saves will start failing"
  fi
  if [ -n "$fu" ] && [ -n "$ft" ] && [ $((fu / 1000)) -ge $((ft / 1000 * 97 / 100)) ]; then
    say "DISK ALERT: files ${fu}/${ft} (>=97%)"
  fi
}

# walltime that always clears the next maintenance window.
# FLOOR, never round: '.0f' once turned 3.56 days into 4-00:00:00, which crossed the
# maintenance window -> every resume sat PENDING (ReqNodeNotAvail) forever.
MAINT_EPOCH=${MAINT_EPOCH:-0}   # unix time of next maintenance start, 0 = none
wt(){ python3 -c "
import math, time
m=$MAINT_EPOCH
d=(m-time.time())/86400-0.3 if m>0 else 4.5
h=max(6,int(math.floor(min(4.5,d)*24)))
print(f'{h//24}-{h%24:02d}:00:00' if h>=24 else f'{h}:00:00')"; }

# ---- singleton guard ----
# Two monitors once ran concurrently and each auto-resumed the same arms, doubling
# the submissions. If an older monitor is already RUNNING, defer to it.
if [ -n "$SLURM_JOB_ID" ]; then
  other=$(squeue -u $USER -h -o "%i %j %T" 2>/dev/null \
          | awk -v me="$SLURM_JOB_ID" '$2=="tcod_monitor" && $3=="RUNNING" && $1+0 < me+0 {print $1}' | head -1)
  if [ -n "$other" ]; then
    say "monitor $SLURM_JOB_ID exiting: older monitor $other is already running"
    exit 0
  fi
fi

# ---- self-renew ----
# On a checkpoint-partition QOS the walltime is capped at ~5 h regardless of what
# --time asks for, so a single monitor job cannot cover a multi-day run. Before this
# one is killed, queue a successor that starts only after we exit (afterany, so
# preemption also triggers it). Combined with the singleton guard this gives
# continuous coverage with no overlap.
RENEWED=0
renew_if_expiring(){
  [ -n "$SLURM_JOB_ID" ] || return 0
  [ "$RENEWED" = 1 ] && return 0
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
    new=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID $CSE/.monitor.sbatch 2>/dev/null)
    if [ -n "$new" ]; then
      say "self-renew: ${secs}s walltime left -> successor job $new (starts when this one ends)"
      RENEWED=1
    fi
  fi
}

say "monitor started (pid $$, job ${SLURM_JOB_ID:-none})"
while true; do
  renew_if_expiring
  {
    echo "=== EXPERIMENT STATUS  $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo
    echo "--- SLURM ---"
    squeue -u $USER -o "%.10i %.20j %.9T %.10M %.13R" 2>/dev/null
    echo
    echo "--- TRAINING PROGRESS ---"
    # Step parsing must NOT require the "monitor.py:NNN]" prefix: full-FT runs log
    # through a different path and write a bare "Step 118:", so a prefix-anchored
    # pattern returned empty for them. Empty then read as "never started" and the
    # resume loop skipped the arm entirely -- it sat dead while the dashboard
    # showed "step -/200".
    for a in "${!ARMS[@]}"; do
      IFS='|' read d cfg part g <<< "${ARMS[$a]}"
      s=$(grep -aoE '(monitor\.py:[0-9]+\] )?Step [0-9]+:' $d/log/trainer.log 2>/dev/null|grep -oE '[0-9]+:$'|tr -d ':'|tail -1)
      tgt=200; case "$a" in opd_v2|dtdd_v2|fullft_opd|fullft_dtdd) tgt=400;; esac
      printf "  %-24s step %s/%s\n" "$a" "${s:-–}" "$tgt"
    done
    echo
    echo "--- MATH RESULTS (held-out) ---"
    python3 - <<'PY' 2>/dev/null
import json,os
p=os.environ["OPD_ROOT"]+"/results/eval_results.jsonl"
if os.path.exists(p):
    d={}
    for l in open(p):
        r=json.loads(l)
        if r.get("metric")=="mean_score":
            d.setdefault((r["arm"],r["step"]),{})[r["benchmark"]]=r["value"]
    for (arm,st),v in sorted(d.items()):
        print(f"  {arm:>12} s{st:<5} " + "  ".join(f"{k}={x:.3f}" for k,x in sorted(v.items())))
PY
    echo
    echo "(auto-resume log: $LOG)"
  } > $STATUS.tmp && mv $STATUS.tmp $STATUS

  # ---- STALL detection: SLURM says RUNNING but the job is a zombie ----
  # A queue-membership check can't see this: when the vLLM engine child dies of
  # SIGBUS the parent blocks forever, so the job stays RUNNING and holds its GPUs
  # while producing nothing (observed: 3 arms x 4 GPUs idle for 3.8 h).
  # Liveness signal = NEWEST mtime among the ckpt-dir logs and the job's .err.
  # Do NOT use the .out alone: it is block-buffered and sits untouched for 7 h on
  # perfectly healthy jobs (measured), which would kill every arm on the first pass.
  # Taking the max only ever makes the check more conservative.
  for a in "${!ARMS[@]}"; do
    [ "${NORESUME:-0}" = 1 ] && continue
    IFS='|' read d cfg part g <<< "${ARMS[$a]}"
    jid=$(squeue -u $USER -h -o "%i %j %T" 2>/dev/null | awk -v n="tcod_${a}" '$2==n && $3=="RUNNING"{print $1}' | head -1)
    [ -z "$jid" ] && continue
    newest=0
    for f in $d/log/trainer.log $d/log/explorer.log $CSE/logs/tcod_${a}_${jid}.err $CSE/logs/tcod_${a}_${jid}.out; do
      [ -f "$f" ] || continue
      m=$(stat -c %Y "$f" 2>/dev/null || echo 0)
      [ "$m" -gt "$newest" ] && newest=$m
    done
    [ "$newest" = 0 ] && continue                     # nothing to judge on -> leave alone
    age=$(( ( $(date +%s) - newest ) / 60 ))
    # FASTKILL: a fatal signal in stderr means the engine child is already dead --
    # the job can only be a zombie. No point waiting the full STALL_MIN (that cost
    # 4 GPU-hours per arm once); 10 min of log silence after the signal is confirmation.
    if grep -aq "Fatal Python error" $CSE/logs/tcod_${a}_${jid}.err 2>/dev/null && [ "$age" -ge 10 ]; then
      say "FASTKILL: $a (job $jid) fatal signal in stderr and logs quiet ${age} min -> scancel"
      scancel $jid 2>/dev/null
      continue
    fi
    if [ "$age" -ge "${STALL_MIN:-60}" ]; then
      say "STALL: $a (job $jid) all logs quiet ${age} min -> declaring zombie, scancel; next pass auto-resumes"
      grep -aoE "Fatal Python error: [A-Za-z ]*|SIGBUS|SIGSEGV" \
        $CSE/logs/tcod_${a}_${jid}.err $CSE/logs/tcod_${a}_${jid}.out 2>/dev/null | tail -1 | sed 's/^/    /' >> $LOG
      scancel $jid 2>/dev/null
    fi
  done

  # ---- crash detection + auto-resume ----
  # STAGGER: every SIGBUS crash we recorded was a LoRA resume launched in a
  # simultaneous batch; jobs started hours apart never hit it. The crash is always
  # a SIGBUS while mmap-loading the freshly written adapter off GPFS ~3-8 min after
  # start -- concurrent startups hammer the same fileset exactly in that window. So
  # delay the k-th resume of a round by k*6 min (--begin), keeping the heavy
  # startup phases disjoint.
  NSUB=0
  for a in "${!ARMS[@]}"; do
    [ "${NORESUME:-0}" = 1 ] && continue
    IFS='|' read d cfg part g <<< "${ARMS[$a]}"
    # EXACT job-name match. A prefix pattern once let fullft_opd match the running
    # tcod_fullft_dtdd and be judged "already running" -- every OPD/DTDD pair
    # silently masked its partner and the dead one was never resumed.
    running=$(squeue -u $USER -h -o "%j" 2>/dev/null | grep -cx "tcod_${a}")
    [ "$running" -gt 0 ] && continue
    step=$(grep -aoE '(monitor\.py:[0-9]+\] )?Step [0-9]+:' $d/log/trainer.log 2>/dev/null|grep -oE '[0-9]+:$'|tr -d ':'|tail -1)
    [ -z "$step" ] && continue                       # never started -> leave alone
    tgt=200; case "$a" in opd_v2|dtdd_v2|fullft_opd|fullft_dtdd) tgt=400;; esac
    [ "$step" -ge "$tgt" ] && continue               # finished
    n=${NRESUME[$a]:-0}
    if [ "$n" -ge "$MAXRESUME" ]; then
      say "$a stopped at step $step but hit resume cap ($MAXRESUME) — needs a human"; continue
    fi
    # The agentic trainer saves explorer/trainer state separately; after a crash the
    # explorer is AHEAD, and on resume the gap steps' experiences are never
    # regenerated -> the trainer times out on an empty batch and the arm dies at the
    # same step forever. Rewind the explorer to the trainer before every resume
    # (no-op when in sync). fix_desync.py belongs to the agentic codebase.
    $CSE/tcod_venv/bin/python $CSE/fix_desync.py --apply "$(basename $d)" >>$LOG 2>&1
    rcfg=${cfg%.yaml}_autoresume.yaml
    sed -e 's/^continue_from_checkpoint: false/continue_from_checkpoint: true/' \
        $CSE/agent_harness/TCOD/$cfg > $CSE/agent_harness/TCOD/$rcfg 2>/dev/null
    # --exclude: one node SIGBUSed the same arm 3/3 times while other nodes ran it
    # fine; queue affinity kept rescheduling the arm onto the bad node.
    jid=$(cd $CSE && sbatch --parsable --job-name=tcod_${a} --account=$MONITOR_ACCOUNT --partition=$part \
      --exclude=${MONITOR_EXCLUDE:-} --begin=now+$((NSUB*6))minutes \
      --gpus=$g --gpus-per-node=$g --cpus-per-task=12 --mem=250G --time=$(wt) \
      --export=ALL,CONFIG=$rcfg \
      $( [[ $a == ws* ]] && echo run_tcod_webshop_l40s.sbatch || echo run_tcod_lora.sbatch ) 2>/dev/null)
    NRESUME[$a]=$((n+1)); NSUB=$((NSUB+1))
    say "AUTO-RESUME $a from step $step -> job $jid (attempt $((n+1))/$MAXRESUME, begin +$(( (NSUB-1)*6 ))min)"
  done

  # ---- math lane (not an "arm": different launcher) ----
  if ! squeue -u $USER -h -o "%j" 2>/dev/null | grep -q "dtddA_taufix"; then
    # Newest ckpt that actually LOADS. A quota-truncated save once left a
    # global_step dir with no zip central directory; resume loaded it, died in
    # 5 min with rc=0, and the resubmit loop burned attempts on a checkpoint that
    # can never load.
    last=$(python3 - <<'PYEOF'
import glob, os, re, zipfile
ds = sorted(glob.glob(os.environ["OPD_ROOT"] + "/checkpoint/dtddA_base_grpo/global_step_[0-9]*"),
            key=lambda d: int(re.search(r"(\d+)$", d).group(1)), reverse=True)
for d in ds:
    shards = glob.glob(d + "/actor/*_world_size_*.pt")
    try:
        ok = bool(shards) and all(zipfile.ZipFile(f).namelist() for f in shards)
    except Exception:
        ok = False
    if ok:
        print(re.search(r"(\d+)$", d).group(1)); break
PYEOF
)
    if [ -n "$last" ] && [ "$last" -lt 279 ] && [ "${NRESUME[math51]:-0}" -lt "$MAXRESUME" ]; then
      # point the arm's submit script at the newest loadable ckpt and resubmit
      sed "s#global_step_[0-9]*#global_step_$last#" $OPD/scripts/train/settings/submit_dtddA_taufix_rs.sh > $OPD/.submit_math51_auto.sh
      sed -i "s/--time=[0-9d:-]*/--time=$(wt)/" $OPD/.submit_math51_auto.sh
      bash $OPD/.submit_math51_auto.sh >/dev/null 2>&1
      NRESUME[math51]=$(( ${NRESUME[math51]:-0} + 1 ))
      say "AUTO-RESUME math51 from step $last (attempt ${NRESUME[math51]}/$MAXRESUME)"
    fi
  fi

  # ---- AUTO-EVAL: submit a held-out eval the moment a new ckpt is fully saved ----
  # Every coverage gap we had came from "train now, evaluate later" (clean ckpts
  # never graded; grid cells deleted by a disk cleanup before evaluation).
  # Save-then-eval => the result lands in the tiny results store first, so pruning
  # the checkpoint later costs nothing. Throttled; ckpt must be shard-complete and
  # quiet for 10 min (see tools/autoeval.py).
  disk_sentinel
  ae=$(python3 $OPD/tools/autoeval.py --apply 2>&1)
  echo "$ae" | grep -q SUBMITTED && say "AUTOEVAL: $(echo "$ae" | grep -c SUBMITTED) new eval(s) submitted"
  echo "$ae" | grep SUBMITTED | sed 's/^/    /' >> $LOG

  # ---- fold newly graded results into the store (figures/tables read from it) ----
  python3 $OPD/tools/harvest.py >/dev/null 2>&1

  # ---- free-space warning ----
  kfree=$(df -BG --output=avail $(dirname $OPD) 2>/dev/null | tail -1 | tr -dc '0-9')
  if [ -n "$kfree" ] && [ "$kfree" -lt 500 ]; then
    say "DISK WARN: only ${kfree}G free (<500G) — consider pruning optimizer states of mid-run ckpts"
  fi
  # ---- inode warning: what actually bit us was the file COUNT, not space.
  # 85k stray temp files exhausted inodes and every write failed with "Disk quota
  # exceeded" while df still showed 1.6T available. Check both lines, always.
  kino=$(hyakstorage --gscratch 2>/dev/null | sed -n "/$(basename $(dirname $OPD))/,/╰/p" \
         | grep -m1 Total | grep -oE '[0-9]+ / [0-9]+ files' | tr -dc '0-9 /')
  kused=${kino%% *}; kcap=${kino##*/ }
  if [ -n "$kused" ] && [ -n "$kcap" ] && [ "$kcap" -gt 0 ]; then
    kirem=$((kcap - kused))
    [ "$kirem" -lt 200000 ] && say "INODE WARN: only ${kirem} inodes left (<200k) — writes will start failing randomly"
  fi
  echo "" >> $STATUS; echo "--- DISK ---" >> $STATUS
  echo "  ${kfree:-?}G free | inodes left ${kirem:-?}" >> $STATUS
  sleep 600
done
