#!/bin/bash
# Resume the §5.1 const-rate arm (DTDD_RATE=0.25 -> adaptive tau pinned at 25% takeover).
# NOTE: this is a §5.1 arm (Qwen3-1.7B-Base <- Qwen3-4B-Base-GRPO), NOT the SFT arm —
# PROGRESS.md previously mislabelled it. Original job 37874697 stopped at step 119
# (35h walltime); ckpt global_step_120 is intact (model+optim, world_size_4).
# Together with the tau=0.2 re-run (~50% takeover) this gives a DOSE-RESPONSE pair.
set -euo pipefail
cd ${OPD_ROOT}
CK=${OPD_ROOT}/checkpoint/dtddA_constrate

COMMON="ALL,\
ACTOR_MODEL_PATH=models/Qwen3-1.7B-Base,\
REWARD_MODEL_PATH=models/Qwen3-4B-Base-GRPO,\
TRAIN_DATASET=datasets/dapo-math-17k.parquet,\
TRAIN_DATASET_NAME=DAPO-Math-17k,\
MAX_RESP_LENGTH=7168,\
MAX_VAL_RESP_LENGTH=7168,\
USER_EXTRA_ARGS=trainer.n_gpus_per_node=2 trainer.default_local_dir=checkpoint/dtddA_constrate actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 trainer.resume_mode=resume_path trainer.resume_from_path=${CK}/global_step_120,\
DTDD_ENABLE=1,DTDD_RATE=0.25,STUDENT_GPU_UTIL=0.45,DTDD_TEACHER_ENGINES=1,DTDD_TEACHER_GPU_UTIL=0.12,\
 WANDB_PROJECT=dtdd_coldstart"

jid=$(sbatch --parsable --job-name=dtddA_constrate_rs \
  --partition=YOUR_GPU_PARTITION --gpus=2 --gpus-per-node=2 --cpus-per-task=12 --mem=200G --time=5-00:00:00 \
  --output=${OPD_ROOT}/logs/%x_%j.out \
  --error=${OPD_ROOT}/logs/%x_%j.err \
  --export="$COMMON" scripts/train/run_dtdd.sbatch)
echo "constrate resume (from step 120, rate=0.25): $jid"
echo "$jid" > .dtddA_constrate_resume_jid
