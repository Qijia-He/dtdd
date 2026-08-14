#!/bin/bash
# COLLAPSE→RESCUE (StableOPD-style), the strategic bet for a genuine DTDD held-out win.
# Student = Qwen2.5-Math-1.5B (StableOPD's student: capable but NON-long-CoT -> prone to the
# length-inflation collapse), Teacher = JustRL-DeepSeek-1.5B (strong, VOCAB-SAFE 151936 so
# DTDD's takeover is valid). DAPO-Math-17k, len 8192, no KL anchor (our default use_kl_loss=False),
# save_freq=10 to resolve the peak-then-drop. JustRL 1.5B teacher pool fits a40, so BOTH arms a40.
#
# Hypothesis: OPD accuracy rises then COLLAPSES (~step 30, length inflation -> truncation/repetition
# dominate). DTDD's divergence-triggered takeover should AUTO-RISE as the student starts to diverge
# and cut the runaway trajectory -> DTDD-final > OPD's post-collapse (and ideally > OPD's pre-collapse
# PEAK, so "just early-stop OPD" can't rebut it).
# STRICT PROTOCOL for eval: OPD at its PEAK and its FINAL step + matched-step DTDD + early-stop-OPD
# as an explicit baseline.
set -euo pipefail
cd ${OPD_ROOT}
[ -f models/Qwen2.5-Math-1.5B/config.json ] || { echo "ERROR: Qwen2.5-Math-1.5B missing"; exit 1; }

BASE="trainer.n_gpus_per_node=4 trainer.save_freq=10 actor_rollout_ref.rollout.gpu_memory_utilization=0.45 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192"
COMMON="ALL,\
ACTOR_MODEL_PATH=models/Qwen2.5-Math-1.5B,\
REWARD_MODEL_PATH=models/JustRL-DeepSeek-1.5B,\
TRAIN_DATASET=datasets/dapo-math-17k.parquet,\
TRAIN_DATASET_NAME=DAPO-Math-17k,\
MAX_RESP_LENGTH=3072,\
MAX_VAL_RESP_LENGTH=3072"

# --- OPD arm (expect collapse) ---
jo=$(sbatch --parsable --job-name=opd_c2 \
  --partition=YOUR_GPU_PARTITION --gpus=4 --gpus-per-node=4 --cpus-per-task=12 --mem=240G --time=7-00:00:00 \
  --output=${OPD_ROOT}/logs/%x_%j.out \
  --error=${OPD_ROOT}/logs/%x_%j.err \
  --export="${COMMON},USER_EXTRA_ARGS=${BASE} trainer.default_local_dir=checkpoint/collapse2_opd,WANDB_PROJECT=dtdd_collapse2" \
  scripts/train/run_opd.sbatch)
echo "OPD  collapse2 submitted: $jo -> logs/opd_c2_${jo}.out"

# --- DTDD arm (rescue; adaptive takeover, JustRL 1.5B teacher pool on a40) ---
jd=$(sbatch --parsable --job-name=dtdd_c2 \
  --partition=YOUR_GPU_PARTITION --gpus=4 --gpus-per-node=4 --cpus-per-task=12 --mem=240G --time=7-00:00:00 \
  --output=${OPD_ROOT}/logs/%x_%j.out \
  --error=${OPD_ROOT}/logs/%x_%j.err \
  --export="${COMMON},USER_EXTRA_ARGS=${BASE} trainer.default_local_dir=checkpoint/collapse2_dtdd,DTDD_ENABLE=1,STUDENT_GPU_UTIL=0.5,DTDD_TEACHER_GPU_UTIL=0.12,WANDB_PROJECT=dtdd_collapse2" \
  scripts/train/run_dtdd.sbatch)
echo "DTDD collapse2 submitted: $jd -> logs/dtdd_c2_${jd}.out"

echo "$jo $jd" > .collapse2_jids
squeue -u USER -h -o "%.10i %.10j %.8T %R" 2>/dev/null | grep -iE "_c2"
