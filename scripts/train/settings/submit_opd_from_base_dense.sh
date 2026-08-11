#!/bin/bash
# §5.1 OPD-from-base RE-RUN with DENSE checkpointing, for the EFFICIENCY curve.
# The original opd_from_base run's intermediate ckpts (20-100) were pruned in a
# disk cleanup, so we only have held-out at 120/200/279 — too sparse to show
# "DTDD reaches accuracy X in fewer steps". This re-run saves every 20 steps so
# we can eval OPD at 20/40/60/80/100/120... and match it against the DTDD arm
# (dtddA_base_grpo), which still has dense ckpts. Identical config to the
# original (same start/teacher/data/length/hparams) — ONLY save_freq + a
# distinct output dir differ, so the curve is a faithful re-measure.
#   student = Qwen3-1.7B-Base   teacher = Qwen3-4B-Base-GRPO   data = DAPO-17k, len 7168
set -euo pipefail
cd ${OPD_ROOT}

COMMON="ALL,\
ACTOR_MODEL_PATH=models/Qwen3-1.7B-Base,\
REWARD_MODEL_PATH=models/Qwen3-4B-Base-GRPO,\
TRAIN_DATASET=datasets/dapo-math-17k.parquet,\
TRAIN_DATASET_NAME=DAPO-Math-17k,\
MAX_RESP_LENGTH=7168,\
MAX_VAL_RESP_LENGTH=7168,\
USER_EXTRA_ARGS=trainer.n_gpus_per_node=4 trainer.save_freq=20 trainer.default_local_dir=checkpoint/opd_from_base_dense actor_rollout_ref.rollout.gpu_memory_utilization=0.4 reward_model.micro_batch_size_per_gpu=8 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192,\
WANDB_PROJECT=dtdd_coldstart"

jid=$(sbatch --parsable \
  --job-name=opd_from_base_dense \
  --partition=YOUR_GPU_PARTITION --gpus=4 --gpus-per-node=4 --cpus-per-task=12 --mem=240G --time=7-00:00:00 \
  --output=${OPD_ROOT}/logs/%x_%j.out \
  --error=${OPD_ROOT}/logs/%x_%j.err \
  --export="$COMMON" \
  run_opd_uw_apptainer.sbatch)
echo "OPD-from-base DENSE submitted: $jid -> logs/opd_from_base_dense_${jid}.out"
echo "$jid" > .opdbase_dense_jid
squeue -h -j "$jid" -o "%i %T %R" 2>/dev/null
