#!/bin/bash
#SBATCH --job-name=sft_code_student
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Build the coding student: SOD's cold-start SFT of Qwen3-1.7B on
# Open-AgentRL-SFT-3K (upstream examples/SOD/run_sft.sh, unmodified: 5 epochs,
# 115 updates on 4 GPUs), then merge the final checkpoint into
# models/Qwen3-1.7B-SOD-SFT, the path code/train.sbatch reads.
#
#   sbatch code/sft.sbatch                      # extra args go to the SFT trainer as hydra overrides
#
# Inputs (relative to DTDD_ROOT unless absolute):
#   BASE_MODEL  models/Qwen3-1.7B  (Qwen/Qwen3-1.7B)
#   SFT_DATA    datasets/code/Open-AgentRL-SFT-3K/full_sft_3k_shuffled_v4.parquet
#               (Gen-Verse/Open-AgentRL-SFT-3K)
set -o pipefail
: "${DTDD_ROOT:?set DTDD_ROOT to this repository}"
: "${CODE_ENV:?set CODE_ENV to the env from code/make_env.sh}"
cd "$DTDD_ROOT"
export PATH="$CODE_ENV/bin:$PATH"
set -eu
mkdir -p logs

abspath() { case "$1" in /*) echo "$1" ;; *) echo "$DTDD_ROOT/$1" ;; esac; }
export MODEL_PATH=$(abspath "${BASE_MODEL:-models/Qwen3-1.7B}")
export TRAIN_DATA=$(abspath "${SFT_DATA:-datasets/code/Open-AgentRL-SFT-3K/full_sft_3k_shuffled_v4.parquet}")
export EVAL_DATA=$TRAIN_DATA
export SAVE_PATH=$(abspath "${SAVE_PATH:-checkpoint/qwen3_1p7b_sft}")
OUT=$(abspath "${OUT:-models/Qwen3-1.7B-SOD-SFT}")
export PROJECT_NAME=dtdd_code EXPERIMENT_NAME=qwen3_1p7b_sft

export TRITON_CACHE_DIR=/tmp/${USER}_${SLURM_JOB_ID}/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/${USER}_${SLURM_JOB_ID}/inductor_cache
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
trap 'rm -rf /tmp/${USER}_${SLURM_JOB_ID} 2>/dev/null || true' EXIT
export WANDB_MODE=${WANDB_MODE:-offline}
export NPROC_PER_NODE=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | wc -l)

# Extra arguments are passed through as hydra overrides (e.g. trainer.total_training_steps=2).
( cd "$DTDD_ROOT/third_party/SOD" && bash examples/SOD/run_sft.sh model.use_liger=True "$@" )

STEP=$(cat "$SAVE_PATH/latest_checkpointed_iteration.txt")
python3 -m verl.model_merger merge --backend fsdp \
  --local_dir "$SAVE_PATH/global_step_$STEP" --target_dir "$OUT"
echo "coding student (SFT step $STEP) -> $OUT"
