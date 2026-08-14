#!/bin/bash
# CLEAN re-run of the §5.1 DTDD-from-base arm, fixing the two bugs that turned the
# previous resume into plain OPD after step 80:
#   (1) previous resume inherited the DEFAULT DTDD_TAU=0.7 -> with d_k p90~0.36 almost
#       nothing triggered -> takeover ~2%. FIX: DTDD_TAU=0.2 (matches the original
#       steps 1-70 that ran ~45% takeover).
#   (2) previous resume auto-loaded the CONTAMINATED global_step_80 (steps 71-80 had
#       been run as plain OPD by a DTDD-disabled bug job). FIX: explicit
#       resume_from_path=global_step_100 (the last CLEAN real-DTDD checkpoint).
# Student Qwen3-1.7B-Base <- teacher Qwen3-4B-Base-GRPO, 7168, 2xA100.
# Resumes at step 70, runs 71->279 as real DTDD, saving into the SAME dir
# (checkpoint/dtddA_base_grpo) at save_freq=20 (matches opd_from_base_dense grid).
set -euo pipefail
cd ${OPD_ROOT}

CKDIR=${OPD_ROOT}/checkpoint/dtddA_base_grpo

COMMON="ALL,\
ACTOR_MODEL_PATH=models/Qwen3-1.7B-Base,\
REWARD_MODEL_PATH=models/Qwen3-4B-Base-GRPO,\
TRAIN_DATASET=datasets/dapo-math-17k.parquet,\
TRAIN_DATASET_NAME=DAPO-Math-17k,\
MAX_RESP_LENGTH=7168,\
MAX_VAL_RESP_LENGTH=7168,\
USER_EXTRA_ARGS=trainer.n_gpus_per_node=2 trainer.default_local_dir=checkpoint/dtddA_base_grpo actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 trainer.resume_mode=resume_path trainer.resume_from_path=${CKDIR}/global_step_100,\
DTDD_ENABLE=1,DTDD_TAU=0.2,STUDENT_GPU_UTIL=0.45,DTDD_TEACHER_ENGINES=1,DTDD_TEACHER_GPU_UTIL=0.12,\
 WANDB_PROJECT=dtdd_coldstart"

jid=$(sbatch --parsable \
  --job-name=dtddA_taufix_rs \
  --partition=YOUR_GPU_PARTITION --gpus=2 --gpus-per-node=2 --cpus-per-task=12 --mem=240G --time=4-00:00:00 \
  --output=${OPD_ROOT}/logs/%x_%j.out \
  --error=${OPD_ROOT}/logs/%x_%j.err \
  --export="$COMMON" \
  scripts/train/run_dtdd.sbatch)
echo "dtddA taufix submitted: $jid -> logs/dtddA_taufix_${jid}.out (resume_from global_step_100, tau=0.2)"
echo "$jid" > .dtddA_taufix_rs_jid
squeue -h -j "$jid" -o "%i %T %R" 2>/dev/null
