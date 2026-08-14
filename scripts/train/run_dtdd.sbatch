#!/bin/bash
#SBATCH --job-name=dtdd_run
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# thunlp/OPD baseline via the verl official container (verlai/verl:vllm011
# as verl_vllm011.sif): torch 2.8+cu128 / vllm 0.11 / flash-attn 2.8.1 --
# the exact stack their conda recipe installs, but immune to the RHEL8
# GLIBC-2.28 problem that breaks the prebuilt flash-attn wheel outside the
# container. Their vendored verl v0.7 + math-verify ride in via PYTHONPATH.
#
# Smoke: sbatch --time=2:00:00 --export=ALL,SMOKE=1 run_opd.sbatch

set -o pipefail
cd ${OPD_ROOT}

SIF=${SIF:?set SIF to your verl+vllm apptainer image}
EXTRA_PKGS=${EXTRA_PKGS:-}

# Node-local, job-scoped caches: keeps per-job churn off a quota-limited shared fs.
JOB_TMP=/tmp/${USER}_${SLURM_JOB_ID}
mkdir -p $JOB_TMP/{triton_cache,triton_home,inductor_cache,tmp,ray,wandb,wandb_data,xdg}
trap "rm -rf $JOB_TMP 2>/dev/null || true" EXIT

_n_gpus=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | wc -l)

# Env for inside the container (apptainer passes host env through).
export TRITON_CACHE_DIR=$JOB_TMP/triton_cache
export TRITON_HOME=$JOB_TMP/triton_home  # autotune cache ignores TRITON_CACHE_DIR; a full home quota kills compiles (Errno 122)
export TORCHINDUCTOR_CACHE_DIR=$JOB_TMP/inductor_cache
export XDG_CACHE_HOME=$JOB_TMP/xdg  # keep vLLM's ~/.cache writes off quota-limited home
export TMPDIR=$JOB_TMP/tmp
export RAY_TMPDIR=$JOB_TMP/ray
export WANDB_DIR=$JOB_TMP/wandb
export WANDB_DATA_DIR=$JOB_TMP/wandb_data  # artifact staging off quota-limited home
export WANDB_MODE=online
export WANDB_PROJECT=${WANDB_PROJECT:-dtdd_cot}
export WANDB_ENTITY="${WANDB_ENTITY}"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((_n_gpus - 1)))
unset ROCR_VISIBLE_DEVICES  # SLURM sets it alongside CVD; verl v0.7 refuses both
export PYTHONPATH=${OPD_ROOT}/verl:${OPD_ROOT}/recipe_dtdd:$EXTRA_PKGS
export DTDD_ENABLE=${DTDD_ENABLE:-1}  # DTDD OFF by default in recipe; enable it here so DTDD arms actually take over
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-models/Qwen3-1.7B-SFT}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-models/Qwen3-4B-Base-GRPO}

if [ "${NO_THINKING:-1}" = "1" ]; then
  EXTRA="+data.apply_chat_template_kwargs.enable_thinking=False trainer.is_plot=False"
else
  EXTRA="trainer.is_plot=False"
fi
if [ "${SMOKE:-0}" = "1" ]; then
  EXTRA="$EXTRA trainer.total_training_steps=2 trainer.save_freq=-1"
fi
export EXTRA_ARGS="$EXTRA ${USER_EXTRA_ARGS:-}"

apptainer exec --nv --bind "${BIND_ROOT:-$PWD}" "$SIF" \
  bash recipe_dtdd/run_dtdd.sh
