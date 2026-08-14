#!/bin/bash
# §5.1 "restart takeover" arm (2026-08-08), motivated by arXiv 2604.13016 §5.1:
# their two-stage result shows the cold-start win comes from teaching the teacher's
# thinking pattern FROM THE PROMPT (200K full-rollout SFT), which mid-trajectory
# suffix takeover provably fails to deliver (our clean fixed-tau arm: train-batch
# +32..57% yet held-out ties OPD on 6 matched cells; takeover 40-55% never decayed).
#
# This arm: identical trigger logic, but on trigger the teacher writes the WHOLE
# trajectory from the prompt (DTDD_MODE=restart, cut=0) = online *targeted* SFT on
# exactly the prompts where the student is most off-pattern.
#
# Const-rate, not fixed tau (DTDD_RATE=0.45, no DTDD_TAU):
#   1. dose control -- full demos close the pattern gap faster, so under fixed tau
#      the dose would self-anneal and confound "mechanism" with "data quantity";
#   2. a fixed tau mis-calibrates once the d_k distribution shifts -- we have seen
#      it drift to ~85% takeover on another domain;
#   3. 0.45 matches the realized takeover of the clean fixed-tau suffix arm, making
#      OPD / suffix-DTDD / restart-DTDD comparable at equal teacher budget.
set -euo pipefail
cd ${OPD_ROOT}

COMMON="ALL,\
ACTOR_MODEL_PATH=models/Qwen3-1.7B-Base,\
REWARD_MODEL_PATH=models/Qwen3-4B-Base-GRPO,\
TRAIN_DATASET=datasets/dapo-math-17k.parquet,\
TRAIN_DATASET_NAME=DAPO-Math-17k,\
MAX_RESP_LENGTH=7168,\
MAX_VAL_RESP_LENGTH=7168,\
USER_EXTRA_ARGS=trainer.n_gpus_per_node=2 trainer.default_local_dir=checkpoint/dtddA_restart actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192,\
DTDD_ENABLE=1,DTDD_RATE=0.45,DTDD_MODE=restart,STUDENT_GPU_UTIL=0.45,DTDD_TEACHER_ENGINES=1,DTDD_TEACHER_GPU_UTIL=0.12,\
 WANDB_PROJECT=dtdd_coldstart"

jid=$(sbatch --parsable \
  --job-name=dtddA_restart \
  --partition=YOUR_GPU_PARTITION --gpus=2 --gpus-per-node=2 --cpus-per-task=12 --mem=240G \
  --time=${WALLTIME:-2-08:00:00} \
  --export="$COMMON" \
  scripts/train/run_dtdd.sbatch)
echo "dtddA_restart submitted: $jid -> logs/dtddA_restart_${jid}.out"
