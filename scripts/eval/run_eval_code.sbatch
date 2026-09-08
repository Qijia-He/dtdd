#!/bin/bash
#SBATCH --job-name=eval_code
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Offline benchmark of code-lane checkpoints: the code analogue of
# scripts/eval/run_eval.sbatch.
#
#   ARM=qwen3_1p7b_code_dtdd STEPS="10 20 30" \
#     sbatch scripts/eval/run_eval_code.sbatch
#
# Why this exists rather than reading the training curves: the training runs
# log val-aux score (a signed reward with partial credit) but not val-core
# accuracy, and the two are not interchangeable -- measured r = 0.30 on one
# arm, whose accuracy sat flat at .215-.228 while its score swung .51 to .58.
# Reading accuracy off a separate fixed harness also makes the number survive
# checkpoint pruning, which is how the other lanes already work.
#
# Sampling is pinned to the arms' training-time validation settings (n=8,
# 12288 response tokens) so these numbers share an axis with the in-training
# curves. Each checkpoint is merged to a Hugging Face directory, scored, and
# the merge deleted again; set KEEP_HF=1 to keep them.
set -o pipefail
: "${OPD_ROOT:?set OPD_ROOT to this repository}"
: "${CODE_ENV:?set CODE_ENV to the env from scripts/env/make_code_env.sh}"
cd "$OPD_ROOT"
export PATH="$CODE_ENV/bin:$PATH"
set -eu

: "${ARM:?set ARM=<checkpoint subdirectory>}"
: "${STEPS:?set STEPS=\"10 20 30\"}"

export TRITON_CACHE_DIR=/tmp/${USER}_${SLURM_JOB_ID}/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/${USER}_${SLURM_JOB_ID}/inductor_cache
export VLLM_CACHE_ROOT=/tmp/${USER}_${SLURM_JOB_ID}/vllm_cache
export TMPDIR=${TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}/tmp}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}/ray}
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" \
         "$TMPDIR" "$RAY_TMPDIR" logs
trap 'rm -rf /tmp/${USER}_${SLURM_JOB_ID} 2>/dev/null || true' EXIT

source scripts/sandbox/acquire_sandbox.sh

export WANDB_MODE=offline
export ENABLE_WANDB=0
export VLLM_USE_V1=1
unset ROCR_VISIBLE_DEVICES
_n=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | wc -l)
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((_n - 1)))

# Four validation sets in one pass. Each isolates a different candidate cause
# for the code lane's train/held-out gap:
#   heldout200_sub    200 unseen problems from the TRAINING distribution with
#                     the same subsampled tests (mean 7) -- same prompt
#                     construction, same reward
#   heldout200_full   the SAME 200 problems with their full suites (mean 45).
#                     sub vs full = how much of the training gain is passing
#                     the subsampled tests rather than solving the problem
#   lcb_v1_fixed      219 stdin problems, 2023-05..2024-03
#   lcb_v6_fixed      112 stdin problems, 2025-01..2025-04
#                     v1 vs v6 = how much of the gap is difficulty, not
#                     distribution shift
DATA=${CODE_DATA_DIR:-datasets/code}
export LIVECODEBENCH=$DATA/lcb_v6_fixed.parquet
# Two GROUPS, not one list. verl concatenates val_files into a single
# dataset, and the two families do not share a schema after their map
# functions: the held-out parquets go through map_fn_skywork and keep
# reward_model.ground_truth as a JSON string, while the lcb_fixed parquets go
# through map_lcb and emit a nested dict. Passing all four at once fails in
# datasets.concatenate_datasets ("features can't be aligned ... ground_truth").
# Grouping by schema costs one extra engine startup per checkpoint.
_held="$DATA/code_slice_heldout200.parquet"
_held_full="$DATA/code_slice_heldout200_fulltests.parquet"
GROUP1="${GROUP1:-$_held $_held_full}"
GROUP2="${GROUP2:-$DATA/lcb_v1_fixed.parquet $DATA/lcb_v6_fixed.parquet}"
_mk() {
  python3 -c "import sys
print('[' + ','.join(repr(p) for p in sys.argv[1:]) + ']')" "$@"
}
_vf1=$(_mk $GROUP1); _vf2=$(_mk $GROUP2)
echo "val group 1 (training-schema held-out): $_vf1"
echo "val group 2 (lcb_fixed):                $_vf2"

# Match the training-time validation.
export N_RESP_PER_PROMPT_VAL=8
export MAX_RESPONSE_LENGTH=12288
export MAX_TURNS=8
# 4 GPUs with TP=4 is exactly one async vLLM replica. More than one replica
# deadlocks on a blocking ray.get inside an async actor on this verl version.
export INFER_TP=4

CK=${CHECKPOINT_ROOT:-checkpoint}/$ARM
STORE=${RESULTS_STORE:-results/eval_results.jsonl}
mkdir -p "$(dirname "$STORE")"

for s in $STEPS; do
  src="$CK/global_step_$s/actor"
  hf="$CK/hf/${ARM}_s$s"
  [ -d "$src" ] || { echo "SKIP s$s: no $src"; continue; }
  if [ ! -f "$hf/model.safetensors" ] && \
     [ ! -f "$hf/model.safetensors.index.json" ]; then
    echo "=== merge $ARM s$s ==="
    python3 -m verl.model_merger merge --backend fsdp \
      --local_dir "$src" --target_dir "$hf" ||
      { echo "MERGE_FAIL s$s"; continue; }
  fi

  log=logs/eval_code_${ARM}_s${s}_${SLURM_JOB_ID}.log
  : > "$log"
  for _g in 1 2; do
    eval "_vf=\$_vf$_g"
    echo "=== eval $ARM s$s  group $_g: $_vf ==="
    MODEL_PATH="$hf" MODEL_TAG="${ARM}_s${s}_g${_g}" \
      bash examples/SOD/eval/run_eval_livecodebench.sh \
        "actor_rollout_ref.rollout.tensor_model_parallel_size=$INFER_TP" \
        "data.val_files=$_vf" \
        2>&1 | tee -a "$log" || echo "EVAL_FAIL s$s group $_g"
  done

  # verl prints the metrics on the console; harvest them into the store.
  ARM="$ARM" STEP="$s" LOG="$log" STORE="$STORE" python3 - <<'PY'
import json
import os
import re

log = os.environ["LOG"]
arm = os.environ["ARM"]
step = int(os.environ["STEP"])
store = os.environ["STORE"]
text = open(log, errors="replace").read()
# Metrics are namespaced by data_source, so one pattern harvests every
# validation set at once.
PATTERN = re.compile(
    r"val-(?:core|aux)/([A-Za-z0-9_]+)/(acc|score)/"
    r"((?:mean|best|worst|maj)@\d+)(?:/mean)?:([-0-9.eE]+)"
)
seen = {}
for data_source, kind, aggregate, value in PATTERN.findall(text):
    # Last occurrence wins.
    seen[(data_source, kind, aggregate)] = float(value)
rows = [
    dict(domain="code", arm=arm, step=step, benchmark=data_source,
         metric=f"{kind}_{aggregate}", value=value, n="8",
         source="run_eval_code.sbatch")
    for (data_source, kind, aggregate), value in sorted(seen.items())
]
if rows:
    with open(store, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print("STORED", len(rows), "rows for", arm, "s%d" % step)
else:
    print("NO METRIC PARSED for", arm, "s%d" % step)
PY

  # The merged copy is a derivative; the row in the store is the durable
  # artifact. Only remove it once this step actually stored something, so a
  # failed evaluation keeps its merge and the retry skips re-merging.
  if [ "${KEEP_HF:-0}" != "1" ] && grep -q "STORED" "$log" 2>/dev/null; then
    rm -rf -- "$hf" && echo "CLEANED merged hf ${ARM}_s$s"
  else
    echo "KEPT merged hf ${ARM}_s$s (nothing stored -- evaluation unfinished)"
  fi
done
echo "EVAL_CODE_DONE $ARM"
