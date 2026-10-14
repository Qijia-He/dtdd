#!/bin/bash
#SBATCH --job-name=dtdd_math
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Math example: the DTDD run of the paper's main design-control table.
#   student Qwen3-1.7B-Base, teacher Qwen3-4B-Base-GRPO, DAPO-Math-17K,
#   7,168-token responses, segment trigger (128-token windows, tau = 0.362),
#   rewind takeover, behaviour cloning on the teacher suffix, 120 updates.
#
#   sbatch math/train.sbatch                      # DTDD
#   DTDD_ENABLE=0 sbatch math/train.sbatch        # plain OPD baseline, same settings
#   SMOKE=1 sbatch --time=2:00:00 math/train.sbatch   # 2-update smoke test
#
# GPU layout (4 GPUs, as in the paper): the student trains on the first two cards
# and two dedicated teacher engines generate takeover suffixes on the last two.
# The minimum is 2 GPUs (one student, one teacher engine). Sharing
# cards between student and teacher has no good operating point (the teacher's
# KV cache either starves generation or the student OOMs on wake-up), so the
# teacher always gets its own GPUs. Any DTDD_* knob can be overridden from the
# environment; see math/dtdd_trainer.py for the full list.

set -o pipefail
cd ${DTDD_ROOT}

SIF=${SIF:?set SIF to your verl+vllm apptainer image}
EXTRA_PKGS=${EXTRA_PKGS:-}

# Node-local, job-scoped caches: keeps per-job churn off a quota-limited shared fs.
JOB_TMP=/tmp/${USER}_${SLURM_JOB_ID}
mkdir -p $JOB_TMP/{triton_cache,triton_home,inductor_cache,tmp,ray,wandb,wandb_data,xdg}
trap "rm -rf $JOB_TMP 2>/dev/null || true" EXIT

_n_gpus=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | wc -l)

# Env for inside the container (apptainer passes host env through).
export TRITON_CACHE_DIR=$JOB_TMP/triton_cache
export TRITON_HOME=$JOB_TMP/triton_home  # autotune cache ignores TRITON_CACHE_DIR; a full home quota kills compiles (Errno 122)
export TORCHINDUCTOR_CACHE_DIR=$JOB_TMP/inductor_cache
export XDG_CACHE_HOME=$JOB_TMP/xdg  # keep vLLM's ~/.cache writes off quota-limited home
export TMPDIR=$JOB_TMP/tmp
export RAY_TMPDIR=$JOB_TMP/ray
export WANDB_DIR=$JOB_TMP/wandb
export WANDB_DATA_DIR=$JOB_TMP/wandb_data  # artifact staging off quota-limited home
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_PROJECT=${WANDB_PROJECT:-dtdd}
export WANDB_ENTITY="${WANDB_ENTITY}"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((_n_gpus - 1)))
unset ROCR_VISIBLE_DEVICES  # SLURM sets it alongside CVD; verl v0.7 refuses both
export PYTHONPATH=${DTDD_ROOT}/verl:${DTDD_ROOT}/math:$EXTRA_PKGS
export DTDD_ENABLE=${DTDD_ENABLE:-1}  # the recipe defaults to OFF (= plain OPD); this example turns it on
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-models/Qwen3-1.7B-Base}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-models/Qwen3-4B-Base-GRPO}   # the teacher
export TRAIN_DATASET=${TRAIN_DATASET:-datasets/dapo-math-17k.parquet}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-7168} MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-7168}
export DTDD_TAU=${DTDD_TAU:-0.362} DTDD_CHUNK=${DTDD_CHUNK:-128}
export DTDD_MIN_PREFIX=${DTDD_MIN_PREFIX:-1} DTDD_MODE=${DTDD_MODE:-rewind}

# GPU split: dedicated teacher engines on the last TEACHER_ENGINES cards
# (2 of 4 GPUs in the paper's runs; 1 of 2 GPUs, the minimum, also works).
TEACHER_ENGINES=${TEACHER_ENGINES:-$([ "$_n_gpus" -ge 4 ] && echo 2 || echo 1)}
if [ "$DTDD_ENABLE" = "1" ]; then STUDENT_GPUS=$(( _n_gpus - TEACHER_ENGINES )); else STUDENT_GPUS=$_n_gpus; fi
export DTDD_TEACHER_ENGINES=$TEACHER_ENGINES DTDD_TEACHER_GPU_OFFSET=$STUDENT_GPUS
export DTDD_TEACHER_GPU_UTIL=${DTDD_TEACHER_GPU_UTIL:-0.80}   # of its own card
export STUDENT_GPU_UTIL=${STUDENT_GPU_UTIL:-0.45}

# Update-pass memory: 80 GB cards fit the full token budget; 44-48 GB cards halve
# it and use Ulysses sequence parallelism (this is what verl_sp_fix.patch is for).
CARD_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "${CARD_MB:-0}" -ge 70000 ]; then TOKENS=${TOKENS:-8192}; SP=${SP:-1}; else TOKENS=${TOKENS:-4096}; SP=${SP:-2}; fi
[ "$STUDENT_GPUS" -lt 2 ] && SP=1
MEMARGS="actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$TOKENS \
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$TOKENS \
actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$TOKENS \
actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP \
actor_rollout_ref.ref.ulysses_sequence_parallel_size=$SP \
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
reward_model.model.fsdp_config.param_offload=True \
reward_model.micro_batch_size_per_gpu=2"
RUN=${RUN:-$([ "$DTDD_ENABLE" = 1 ] && echo dtdd_math || echo opd_math)}

if [ "${NO_THINKING:-1}" = "1" ]; then
  EXTRA="+data.apply_chat_template_kwargs.enable_thinking=False trainer.is_plot=False"
else
  EXTRA="trainer.is_plot=False"
fi
EXTRA="$EXTRA trainer.n_gpus_per_node=$STUDENT_GPUS trainer.total_training_steps=${STEPS:-120} \
trainer.save_freq=20 trainer.resume_mode=auto trainer.experiment_name=$RUN \
trainer.default_local_dir=${CKPT_DIR:-checkpoint/$RUN} $MEMARGS"
if [ "${SMOKE:-0}" = "1" ]; then
  EXTRA="$EXTRA trainer.total_training_steps=2 trainer.save_freq=-1"
fi
export EXTRA_ARGS="$EXTRA ${USER_EXTRA_ARGS:-}"

apptainer exec --nv --bind "${BIND_ROOT:-$PWD}" "$SIF" \
  bash math/run_dtdd.sh
