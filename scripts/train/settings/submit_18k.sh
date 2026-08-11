#!/bin/bash
# Revive the 18K length-sweep point (4th length: 7K/12K/15K/18K) on the now-free
# A100s. Key insight: ppo_max_token_len_per_gpu defaults to 32768, which caps the
# per-microbatch token count REGARDLESS of individual sequence length. An 18K
# response (~19456 w/ prompt) < 32768 -> one seq per microbatch, so activation
# memory ~= the 15K run that already fit on 4xA100-80G. The a40 OOM at 18K was the
# 48G card being too small for a 32768-token fp32 microbatch; A100-80G is not.
# Clones the proven-good 15K B-arm env; only MAX_RESP_LENGTH changes 15360->18432.
# Pair B: student R1-Distill-1.5B <- teacher JustRL-1.5B.  Both arms -> 4xA100.
set -euo pipefail
cd ${OPD_ROOT}

COMMON="ALL,\
ACTOR_MODEL_PATH=models/DeepSeek-R1-Distill-Qwen-1.5B,\
REWARD_MODEL_PATH=models/JustRL-DeepSeek-1.5B,\
MAX_RESP_LENGTH=18432,\
MAX_VAL_RESP_LENGTH=7168,\
USER_EXTRA_ARGS=trainer.n_gpus_per_node=4,\
WANDB_PROJECT=dtdd_lensweep"

A100="--partition=YOUR_GPU_PARTITION --gpus=4 --gpus-per-node=4 --cpus-per-task=12 --mem=240G --time=7-00:00:00"

jid_opd=$(sbatch --parsable --job-name=opdB_18k $A100 \
  --output=${OPD_ROOT}/logs/%x_%j.out \
  --error=${OPD_ROOT}/logs/%x_%j.err \
  --export="$COMMON" run_opd_uw_apptainer.sbatch)
echo "OPD  18K submitted: $jid_opd -> logs/opdB_18k_${jid_opd}.out"

jid_dtdd=$(sbatch --parsable --job-name=dtddB_18k $A100 \
  --output=${OPD_ROOT}/logs/%x_%j.out \
  --error=${OPD_ROOT}/logs/%x_%j.err \
  --export="$COMMON" run_dtdd_uw_apptainer.sbatch)
echo "DTDD 18K submitted: $jid_dtdd -> logs/dtddB_18k_${jid_dtdd}.out"

echo "$jid_opd $jid_dtdd" > .len18k_jids
squeue -u USER -o "%.10i %.12j %.9T %R" | grep -E "opdB_18k|dtddB_18k" || true
