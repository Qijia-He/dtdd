#!/bin/bash
#SBATCH --job-name=dtdd_code
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=280G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Train one code-lane arm (DTDD or the matched OPD baseline).
#
#   ARM=dtdd sbatch scripts/train/run_dtdd_code.sbatch      # tau = 0.50
#   ARM=dtdd DTDD_TAU=0.26 EXPERIMENT_NAME=code_dtdd_t026 \
#       sbatch scripts/train/run_dtdd_code.sbatch           # the high-dose arm
#   ARM=opd  sbatch scripts/train/run_dtdd_code.sbatch      # the baseline
#
# Anything else is passed through as a hydra override. The 4 GPUs split
# 2 (student) + 2 (frozen teacher pool); see recipe/dtdd/run_dtdd_code.sh.
#
# Unlike the math and agentic lanes, this one does not use the container:
# the SOD base pins vllm<=0.9.1, an older stack than the image carries. It
# runs in the environment built by scripts/env/make_code_env.sh; point
# CODE_ENV at it.
#
# Checkpoints resume automatically (trainer.resume_mode=auto): after a
# timeout or preemption, resubmit the same command. Chain jobs with
# --dependency=afterany rather than retrying inside one allocation -- a
# crashed engine's memory is not released, and the student then cannot even
# allocate its KV cache on resume.
set -o pipefail
: "${OPD_ROOT:?set OPD_ROOT to this repository}"
: "${CODE_ENV:?set CODE_ENV to the env from scripts/env/make_code_env.sh}"
cd "$OPD_ROOT"
export PATH="$CODE_ENV/bin:$PATH"
set -eu

: "${ARM:?set ARM=dtdd or ARM=opd}"

# Caches on node-local storage: a full or quota-bound shared filesystem
# turns a kernel compile into a silent job-wide hang (see the overlay's
# constants_ppo hunk).
export TRITON_CACHE_DIR=/tmp/${USER}_${SLURM_JOB_ID}/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/${USER}_${SLURM_JOB_ID}/inductor_cache
export VLLM_CACHE_ROOT=/tmp/${USER}_${SLURM_JOB_ID}/vllm_cache
export TMPDIR=${TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}/tmp}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}/ray}
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" \
         "$TMPDIR" "$RAY_TMPDIR" logs
trap 'rm -rf /tmp/${USER}_${SLURM_JOB_ID} 2>/dev/null || true' EXIT

# A code execution sandbox is mandatory: every training response is graded by
# running it. Reuses SANDBOX_FUSION_URL when set, otherwise starts a
# dedicated instance for this job and cancels it on exit.
source scripts/sandbox/acquire_sandbox.sh

# Ray renumbers the job's GPUs 0..N-1 inside the allocation.
_n_gpus=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | wc -l)
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((_n_gpus - 1)))
export VLLM_USE_V1=1
unset ROCR_VISIBLE_DEVICES

export STUDENT_MODEL_PATH=${STUDENT_MODEL_PATH:-models/Qwen3-1.7B-SFT}
export TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-models/SOD-GRPO_teacher-4B}
export OPEN_AGENT_RL=${OPEN_AGENT_RL:-datasets/code/code_slice_train.parquet}
export LIVECODEBENCH=${LIVECODEBENCH:-datasets/code/lcb_v6_fixed.parquet}

# The two arms differ only in the trigger threshold and the random-trigger
# probability; see recipe/dtdd/run_dtdd_code.sh for why the baseline keeps
# the DTDD code path.
if [ "$ARM" = "dtdd" ]; then
  export DTDD_TAU=${DTDD_TAU:-0.5}
  export DTDD_EPSILON_RANDOM=${DTDD_EPSILON_RANDOM:-0.2}
else
  export DTDD_TAU=${DTDD_TAU:-999}
  export DTDD_EPSILON_RANDOM=${DTDD_EPSILON_RANDOM:-0}
fi
export PROJECT_NAME=${PROJECT_NAME:-dtdd_code}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_1p7b_code_${ARM}}
export DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-checkpoint/$EXPERIMENT_NAME}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-wandb_logs}
mkdir -p "$WANDB_DIR"

bash recipe/dtdd/run_dtdd_code.sh "$@"
