#!/bin/bash
#SBATCH --job-name=thunlp_grpo
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

# RL (GRPO) baseline: thunlp/OPD's grpo.sh with the student swapped in
# (their default trains the 4B teacher; our baseline trains Qwen3-1.7B-SFT
# with rule-based math reward, no teacher). Same container stack as the
# OPD baseline. Dataset stays their grpo default (dapo-math-17k-processed:
# bare problems -> model answers in \boxed{}, which is the only format the
# ttrl_math reward extracts; the instruction-prefixed parquet would yield
# all-zero rewards and no GRPO signal).
#
# Smoke: sbatch --time=2:00:00 --export=ALL,SMOKE=1 run_grpo.sbatch

set -o pipefail
cd ${OPD_ROOT}

SIF=${SIF:?set SIF to your verl+vllm apptainer image}
EXTRA_PKGS=${EXTRA_PKGS:-}

# Node-local, job-scoped caches (gscratch file-count quota protection).
JOB_TMP=/tmp/${USER}_${SLURM_JOB_ID}
mkdir -p $JOB_TMP/{triton_cache,inductor_cache,tmp,ray,wandb,wandb_data}
trap "rm -rf $JOB_TMP 2>/dev/null || true" EXIT

_n_gpus=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | wc -l)

export TRITON_CACHE_DIR=$JOB_TMP/triton_cache
export TORCHINDUCTOR_CACHE_DIR=$JOB_TMP/inductor_cache
export TMPDIR=$JOB_TMP/tmp
export RAY_TMPDIR=$JOB_TMP/ray
export WANDB_DIR=$JOB_TMP/wandb
export WANDB_DATA_DIR=$JOB_TMP/wandb_data  # artifact staging off quota-limited home
export WANDB_MODE=online
export WANDB_PROJECT=${WANDB_PROJECT:-dtdd_cot}
export WANDB_ENTITY="${WANDB_ENTITY}"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((_n_gpus - 1)))
unset ROCR_VISIBLE_DEVICES  # SLURM sets it alongside CVD; verl v0.7 refuses both
export PYTHONPATH=${OPD_ROOT}/verl:$EXTRA_PKGS
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-models/Qwen3-1.7B-SFT}

# grpo.sh hardcodes n_gpus_per_node=8; EXTRA_ARGS comes last so hydra
# takes these overrides. enable_thinking=False is already in grpo.sh.
EXTRA="trainer.is_plot=False trainer.n_gpus_per_node=$_n_gpus actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 trainer.test_freq=-1"
if [ "${SMOKE:-0}" = "1" ]; then
  EXTRA="$EXTRA trainer.total_training_steps=2 trainer.save_freq=-1"
fi
export EXTRA_ARGS="$EXTRA ${USER_EXTRA_ARGS:-}"

apptainer exec --nv --bind "${BIND_ROOT:-$PWD}" "$SIF" \
  bash grpo_uw.sh
