#!/bin/bash
# StableOPD collapse→rescue repro (§7 item 5) — the durable-accuracy bet in DTDD's proven lane.
# Student = DeepSeek-R1-Distill-Qwen-1.5B (capable, long-CoT: real accuracy to lose),
# Teacher = DeepSeek-R1-Distill-Qwen-7B (StableOPD's teacher; vocab 152064 > student 151936,
# handled by the sample_vocab_cap mask in recipe_dtdd — teacher sampling capped to student ids).
# len 7168 (collapse2's fatal 3072 truncation avoided), save_freq=10 (dense: resolves the
# peak-then-drop AND provides the early-stop-OPD baseline).
# WALLTIME: must END before the next maintenance reservation or the job defers past it
# (check `scontrol show reservation`; Aug11 09:00 maintenance -> 2026-08-05 submits used
# 5-18:00:00 via scontrol update). Default below is 6d for post-maintenance reuse.
#
# Hypothesis: OPD accuracy peaks then COLLAPSES (length-inflation: reverse-KL rewards
# repetition -> runaway length -> truncation dominates, StableOPD ~step30). DTDD's divergence
# tripwire auto-rises as the student drifts and cuts the runaway -> DTDD final >= OPD peak.
# STRICT PROTOCOL: eval OPD at PEAK + FINAL, matched-step DTDD, early-stop-OPD as baseline.
#
# Usage:
#   bash submit_stableopd.sh            # phase 1: tau-calibration dry-run + OPD arm
#   bash submit_stableopd.sh dtdd 0.35  # phase 2: DTDD arm at the calibrated tau
set -euo pipefail
cd ${OPD_ROOT}
[ -f models/DeepSeek-R1-Distill-Qwen-1.5B/config.json ] || { echo "ERROR: student missing"; exit 1; }
[ -f models/DeepSeek-R1-Distill-Qwen-7B/config.json ] || { echo "ERROR: teacher missing"; exit 1; }

# save_freq=20 (user decision 2026-08-05): full saves incl. optimizer, ~14 ckpts/arm
# (~530GB both arms) — disk at 98% ruled out save_freq=10's ~1.05TB.
BASE="trainer.n_gpus_per_node=4 trainer.save_freq=20 reward_model.micro_batch_size_per_gpu=12 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192"
COMMON="ALL,\
ACTOR_MODEL_PATH=models/DeepSeek-R1-Distill-Qwen-1.5B,\
REWARD_MODEL_PATH=models/DeepSeek-R1-Distill-Qwen-7B,\
TRAIN_DATASET=datasets/dapo-math-17k.parquet,\
TRAIN_DATASET_NAME=DAPO-Math-17k,\
MAX_RESP_LENGTH=7168,\
MAX_VAL_RESP_LENGTH=7168"
SBATCH_GPU4="--partition=YOUR_GPU_PARTITION --gpus=4 --gpus-per-node=4 --cpus-per-task=12 --mem=240G --time=6-00:00:00 \
  --output=${OPD_ROOT}/logs/%x_%j.out \
  --error=${OPD_ROOT}/logs/%x_%j.err"

MODE=${1:-launch}

if [ "$MODE" = "launch" ]; then
  # --- tau calibration: DTDD dry-run, 10 steps, no takeover, no teacher pool, no ckpts ---
  jc=$(sbatch --parsable --job-name=so_cal $SBATCH_GPU4 \
    --export="${COMMON},DTDD_ENABLE=1,DTDD_DRY_RUN=1,DTDD_DK_DUMP=${OPD_ROOT}/reports/stableopd_dk,\
USER_EXTRA_ARGS=${BASE} trainer.default_local_dir=checkpoint/stableopd_cal trainer.total_training_steps=10 trainer.save_freq=-1,WANDB_PROJECT=dtdd_stableopd" \
    scripts/train/run_dtdd.sbatch)
  echo "CAL  submitted: $jc -> logs/so_cal_${jc}.out (read dtdd/would_trigger@tau lines)"

  # --- OPD arm (expect collapse; needs no tau) ---
  jo=$(sbatch --parsable --job-name=so_opd $SBATCH_GPU4 \
    --export="${COMMON},USER_EXTRA_ARGS=${BASE} trainer.default_local_dir=checkpoint/stableopd_opd,WANDB_PROJECT=dtdd_stableopd" \
    scripts/train/run_opd.sbatch)
  echo "OPD  submitted: $jo -> logs/so_opd_${jo}.out"
  echo "$jc $jo" > .stableopd_jids

elif [ "$MODE" = "opd" ]; then
  jo=$(sbatch --parsable --job-name=so_opd $SBATCH_GPU4 \
    --export="${COMMON},USER_EXTRA_ARGS=${BASE} trainer.default_local_dir=checkpoint/stableopd_opd,WANDB_PROJECT=dtdd_stableopd" \
    scripts/train/run_opd.sbatch)
  echo "OPD  submitted: $jo -> logs/so_opd_${jo}.out"
  echo "$jo" >> .stableopd_jids

elif [ "$MODE" = "dtdd" ]; then
  TAU=${2:?Usage: submit_stableopd.sh dtdd <tau>}
  # 6 GPUs: 4 student (trainer.n_gpus_per_node=4) + 2 dedicated teacher engines on GPUs 4-5
  # (the proven dtddSFT layout: offset=4, util=0.8; 7B bf16 ~15GB fits an l40s at 0.8 easily).
  jd=$(sbatch --parsable --job-name=so_dtdd \
    --partition=YOUR_GPU_PARTITION --gpus=6 --gpus-per-node=6 --cpus-per-task=16 --mem=300G --time=6-00:00:00 \
    --output=${OPD_ROOT}/logs/%x_%j.out \
    --error=${OPD_ROOT}/logs/%x_%j.err \
    --export="${COMMON},DTDD_ENABLE=1,DTDD_TAU=${TAU},DTDD_TEACHER_ENGINES=2,DTDD_TEACHER_GPU_OFFSET=4,DTDD_TEACHER_GPU_UTIL=0.8,\
USER_EXTRA_ARGS=${BASE} trainer.default_local_dir=checkpoint/stableopd_dtdd,WANDB_PROJECT=dtdd_stableopd" \
    scripts/train/run_dtdd.sbatch)
  echo "DTDD submitted: $jd (tau=${TAU}) -> logs/so_dtdd_${jd}.out"
  echo "$jd" >> .stableopd_jids
else
  echo "Unknown mode: $MODE"; exit 1
fi
squeue -u USER -h -o "%.10i %.10j %.8T %R" 2>/dev/null | grep -E "so_" || true
