#!/bin/bash
#SBATCH --job-name=eval_generic
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Generic verl/math held-out eval: merge FSDP checkpoint(s) -> vLLM gen -> grade.
# Driven entirely by env so autoeval.py can point it at any arm:
#   ARM=<eval name prefix>  CKDIR=<abs checkpoint dir>  STEPS="20 40 ..."
# Results land in scripts/val/eval/justrl_eval_outputs/<ARM>_s<STEP>/ and are picked up
# by results/harvest.py into the checkpoint-free results store.
set -o pipefail
cd ${OPD_ROOT}
SIF=${SIF:?set SIF to your verl+vllm apptainer image}
EXTRA_PKGS=${EXTRA_PKGS:-}
JOB_TMP=/tmp/${USER}_${SLURM_JOB_ID}
mkdir -p $JOB_TMP/{tmp,triton,inductor,xdg}
trap "rm -rf $JOB_TMP 2>/dev/null || true" EXIT
export TMPDIR=$JOB_TMP/tmp TRITON_CACHE_DIR=$JOB_TMP/triton
export TORCHINDUCTOR_CACHE_DIR=$JOB_TMP/inductor XDG_CACHE_HOME=$JOB_TMP/xdg
_n=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | wc -l)
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((_n-1))); unset ROCR_VISIBLE_DEVICES

: ${ARM:?}; : ${CKDIR:?}; : ${STEPS:?}
D=$CKDIR
MODELS=""
for s in $STEPS; do
  tgt="$D/hf/${ARM}_s$s"
  if [ ! -f "$tgt/model.safetensors" ] && [ ! -f "$tgt/model.safetensors.index.json" ]; then
    echo "=== merge $ARM step $s ==="
    PYTHONPATH=${OPD_ROOT}/verl:$EXTRA_PKGS \
    apptainer exec --bind "${BIND_ROOT:-$PWD}" "$SIF" \
      python3 -m verl.model_merger merge --backend fsdp \
        --local_dir "$D/global_step_$s/actor" --target_dir "$tgt" || { echo "MERGE_FAIL $s"; continue; }
  fi
  MODELS="${MODELS:+$MODELS:}$tgt"
done
export EVAL_MODELS="$MODELS"
echo "MERGED. EVAL_MODELS=$EVAL_MODELS"

cd ${OPD_ROOT}/scripts/eval
export PYTHONPATH=$EXTRA_PKGS
apptainer exec --nv --bind "${BIND_ROOT:-$PWD}" "$SIF" bash -c "python3 gen_vllm.py --disable-thinking"

for s in $STEPS; do
  name="${ARM}_s$s"
  n=$(ls justrl_eval_outputs/$name/*.jsonl 2>/dev/null | wc -l)
  if [ "$n" = 3 ]; then EVAL_NAME=$name apptainer exec --bind "${BIND_ROOT:-$PWD}" "$SIF" python3 grade.py
  else echo "SKIP grading $name: $n/3 files"; fi
done
echo "EVAL_DONE"
