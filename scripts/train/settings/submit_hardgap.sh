#!/bin/bash
# Launch the hard-gap acc-bet arms: OPD vs DTDD on the clean hard subset
# (student pass@4==0 AND teacher pass>=0.5, measured @7168 tok on DAPO-Math-17k).
# Pair B: student = DeepSeek-R1-Distill-Qwen-1.5B  <-  teacher = JustRL-DeepSeek-1.5B.
# Length = 7168 (the length the subset was DEFINED at, so "teacher can solve" holds).
# Clones the proven-good 7K B-arm env; only TRAIN_DATASET changes.
#
# Usage:  bash submit_hardgap.sh <subset_parquet>   # default datasets/dapo_hardgap.parquet
set -euo pipefail
cd ${OPD_ROOT}

SUBSET=${1:-datasets/dapo_hardgap.parquet}
[ -f "$SUBSET" ] || { echo "ERROR: subset $SUBSET not found — tagging not done?"; exit 1; }
NROWS=$(apptainer exec --bind "${BIND_ROOT:-$PWD}" ${SIF} \
        python -c "import pandas as pd; print(len(pd.read_parquet('$SUBSET')))" 2>/dev/null || echo "?")
echo "subset $SUBSET rows=$NROWS"

COMMON_ENV="ALL,\
ACTOR_MODEL_PATH=models/DeepSeek-R1-Distill-Qwen-1.5B,\
REWARD_MODEL_PATH=models/JustRL-DeepSeek-1.5B,\
TRAIN_DATASET=$SUBSET,\
TRAIN_DATASET_NAME=dapo_hardgap,\
USER_EXTRA_ARGS=trainer.n_gpus_per_node=4 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 trainer.total_epochs=10,\
DTDD_ENABLE=1,STUDENT_GPU_UTIL=0.5,DTDD_TEACHER_GPU_UTIL=0.12,\
 WANDB_PROJECT=dtdd_hardgap"

# OPD arm (the one expected to get weak signal on these hard prompts)
jid_opd=$(sbatch --parsable \
  --job-name=opd_hardgap \
  --partition=YOUR_GPU_PARTITION --gpus=4 --gpus-per-node=4 --cpus-per-task=12 --mem=240G --time=7-00:00:00 \
  --output=${OPD_ROOT}/logs/%x_%j.out \
  --error=${OPD_ROOT}/logs/%x_%j.err \
  --export="$COMMON_ENV" \
  scripts/train/run_opd.sbatch)
echo "OPD  hard-gap submitted: $jid_opd  -> logs/opd_hardgap_${jid_opd}.out"

# DTDD arm (teacher takeover injects the correct trajectory the student can't sample)
jid_dtdd=$(sbatch --parsable \
  --job-name=dtdd_hardgap \
  --partition=YOUR_GPU_PARTITION --gpus=4 --gpus-per-node=4 --cpus-per-task=12 --mem=240G --time=7-00:00:00 \
  --output=${OPD_ROOT}/logs/%x_%j.out \
  --error=${OPD_ROOT}/logs/%x_%j.err \
  --export="$COMMON_ENV" \
  scripts/train/run_dtdd.sbatch)
echo "DTDD hard-gap submitted: $jid_dtdd -> logs/dtdd_hardgap_${jid_dtdd}.out"

echo "$jid_opd $jid_dtdd" > .hardgap_jids
