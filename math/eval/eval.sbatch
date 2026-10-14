#!/bin/bash
#SBATCH --job-name=eval_math
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
# Math held-out eval: merge FSDP checkpoint(s) -> vLLM generation -> grading on
# AIME24, AIME25 and AMC23 (16 samples per problem, temperature 0.7, top-p 0.95).
#   ARM=dtdd_math CKDIR=$DTDD_ROOT/checkpoint/dtdd_math STEPS="60 120" sbatch math/eval/eval.sbatch
# Results land in math/eval/outputs/<ARM>_s<STEP>/grading_results.json.
set -o pipefail
cd ${DTDD_ROOT}
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
    PYTHONPATH=${DTDD_ROOT}/verl:$EXTRA_PKGS \
    apptainer exec --bind "${BIND_ROOT:-$PWD}" "$SIF" \
      python3 -m verl.model_merger merge --backend fsdp \
        --local_dir "$D/global_step_$s/actor" --target_dir "$tgt" || { echo "MERGE_FAIL $s"; continue; }
  fi
  MODELS="${MODELS:+$MODELS:}$tgt"
done
export EVAL_MODELS="$MODELS"
echo "MERGED. EVAL_MODELS=$EVAL_MODELS"

cd ${DTDD_ROOT}/math/eval
export PYTHONPATH=$EXTRA_PKGS
# benchmark parquets come from the upstream base that setup.sh cloned
export EVAL_DATA_DIR=${EVAL_DATA_DIR:-${DTDD_ROOT}/third_party/OPD/scripts/val/data}
apptainer exec --nv --bind "${BIND_ROOT:-$PWD}" "$SIF" bash -c "python3 gen_vllm.py --disable-thinking"

for s in $STEPS; do
  name="${ARM}_s$s"
  n=$(ls outputs/$name/*.jsonl 2>/dev/null | wc -l)
  if [ "$n" = 3 ]; then EVAL_NAME=$name apptainer exec --bind "${BIND_ROOT:-$PWD}" "$SIF" python3 grade.py
  else echo "SKIP grading $name: $n/3 files"; fi
done
echo "EVAL_DONE"
