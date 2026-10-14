#!/bin/bash
# The hydra override bundle behind every code-lane number in the paper.
#
# Defaults reproduce the paper's arms on 4 GPUs: the student trains on the
# first two (TP=2) and the frozen teacher pool takes the other two (dp=2,
# TP=1) through DTDD_TEACHER_GPU_OFFSET=2. Sharing devices between the two
# has no working operating point -- see recipe/dtdd/dtdd_ray_trainer.py.
#
# Both paper arms run through this one script; they differ in two values:
#
#   DTDD (tau=0.50)   DTDD_TAU=0.5   DTDD_EPSILON_RANDOM=0.2
#   OPD baseline      DTDD_TAU=999   DTDD_EPSILON_RANDOM=0
#
# The baseline keeps dtdd.enable=true so that both arms share the rollout
# path, the teacher engine and the loss code exactly; a threshold of 999 with
# no random triggers simply never fires, leaving plain per-token OPD on the
# whole episode. STEPWISE_ENABLE=False is the other half of "plain": the w_k
# step weighting is SOD's contribution, not vanilla OPD, and both arms run
# without it so the code lane's baseline means the same thing as the math
# and agentic lanes'.
#
# Run from the SOD checkout root, with a sandbox endpoint already exported:
#
#   SANDBOX_FUSION_URL=http://<host>:8080/run_code \
#   STUDENT_MODEL_PATH=... TEACHER_MODEL_PATH=... \
#   OPEN_AGENT_RL=.../code_slice_train.parquet \
#   bash recipe/dtdd/run_dtdd_code.sh <extra hydra overrides>
#
# code/train.sbatch wraps this with the cluster plumbing.
set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
# recipe.* and the agent-loop _target_ must be importable inside Ray workers.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# Put the teacher engines on the GPUs the student does not use.
export DTDD_TEACHER_GPU_OFFSET=${DTDD_TEACHER_GPU_OFFSET:-2}

# ---- Data ----
: "${OPEN_AGENT_RL:?set OPEN_AGENT_RL to code_slice_train.parquet}"
: "${LIVECODEBENCH:?set LIVECODEBENCH to an lcb_*_fixed.parquet}"
train_files=${TRAIN_FILES:-"['$OPEN_AGENT_RL']"}
val_files=${VAL_FILES:-"['$LIVECODEBENCH']"}

# ---- Models ----
student_model_path=${STUDENT_MODEL_PATH:?set STUDENT_MODEL_PATH}
teacher_model_path=${TEACHER_MODEL_PATH:?set TEACHER_MODEL_PATH}

# ---- Tools ----
_default_tool_config=recipe/demystify/sandbox_fusion_tool_config.yaml
tool_config_path=${TOOL_CONFIG_PATH:-$_default_tool_config}

# ---- Naming ----
project_name=${PROJECT_NAME:-dtdd_code}
experiment_name=${EXPERIMENT_NAME:-qwen3_1p7b_code_dtdd}
default_local_dir=${DEFAULT_LOCAL_DIR:-./checkpoint/$experiment_name}

# ---- DTDD (see recipe/dtdd/config/dtdd_trainer.yaml for what each does) ----
dtdd_enable=${DTDD_ENABLE:-True}
dtdd_tau=${DTDD_TAU:-0.5}
dtdd_epsilon_random=${DTDD_EPSILON_RANDOM:-0.2}
dtdd_takeover_mode=${DTDD_TAKEOVER_MODE:-rewind}
dtdd_dry_run=${DTDD_DRY_RUN:-False}
dtdd_grpo_prefix_coef=${DTDD_GRPO_PREFIX_COEF:-0.0}   # 0 = pure distillation
dtdd_suffix_sft_coef=${DTDD_SUFFIX_SFT_COEF:-1.0}
dtdd_suffix_loss_type=${DTDD_SUFFIX_LOSS_TYPE:-ce}    # ce | kl
dtdd_drop_failed_suffix=${DTDD_DROP_FAILED_SUFFIX:-False}
teacher_infer_tp=${TEACHER_INFER_TP:-1}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.8}
teacher_max_num_seqs=${TEACHER_MAX_NUM_SEQS:-64}
student_gpu_mem_util=${STUDENT_GPU_MEM_UTIL:-0.45}

# ---- GRPO / reward (as in the upstream SOD recipe) ----
adv_estimator=grpo
use_kl_in_reward=False
teacher_kl_coef=0.002

use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode=token-mean
reward_manager=dapo
ol_enable=True
ol_buffer_len=$((1024 * 1))
ol_pen=1.0

enable_filter_groups=False
filter_groups_metric=seq_reward
filter_std_threshold=0.0

# ---- OPD term. stepwise_enable off = vanilla per-token OPD (see header) ----
token_kl_reg_enable=True
stepwise_enable=${STEPWISE_ENABLE:-False}
stepwise_epsilon=1e-6
stepwise_delta=0.2
stepwise_opd_coef=1.0
token_kl_gamma=1.0
token_kl_beta_min=0.0
token_kl_beta_max=0.10

# ---- Training ----
max_turns=${MAX_TURNS:-8}
max_prompt_length=2560
max_response_length=${MAX_RESPONSE_LENGTH:-12288}
actor_lr=1e-6

train_batch_size=${TRAIN_BATCH_SIZE:-64}
ppo_mini_batch_size=16
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}
n_resp_per_prompt_val=${N_RESP_PER_PROMPT_VAL:-8}

n_gpus_per_node=${N_GPUS_PER_NODE:-2}
infer_tp=${INFER_TP:-2}    # rollout TP must equal the student world size
train_sp=${TRAIN_SP:-2}
offload=${FSDP_OFFLOAD:-True}

# 8192, not (prompt + response). The vocabulary logits gradient inside the
# update's backward pass is the peak allocation of the whole step, and the
# full-length value overflows a 48 GB card. Legality: this times the Ulysses
# sequence-parallel degree must be at least the longest sequence.
actor_max_len=${ACTOR_MAX_TOKEN_LEN_PER_GPU:-8192}
logprob_max_len=$((actor_max_len * 4))

# Absolute, so Ray workers resolve it regardless of working directory.
agent_loop_cfg=$PWD/recipe/dtdd/config/agent_loop.yaml

# ---- Launch ----
python3 -m recipe.dtdd.main_dtdd \
    dtdd.enable=$dtdd_enable \
    dtdd.tau=$dtdd_tau \
    dtdd.epsilon_random=$dtdd_epsilon_random \
    dtdd.takeover_mode=$dtdd_takeover_mode \
    dtdd.dry_run=$dtdd_dry_run \
    dtdd.grpo_prefix_coef=$dtdd_grpo_prefix_coef \
    dtdd.suffix_sft_coef=$dtdd_suffix_sft_coef \
    dtdd.suffix_loss_type=$dtdd_suffix_loss_type \
    dtdd.drop_failed_suffix=$dtdd_drop_failed_suffix \
    dtdd.teacher.model_path=$teacher_model_path \
    dtdd.teacher.tensor_model_parallel_size=$teacher_infer_tp \
    dtdd.teacher.gpu_memory_utilization=$teacher_gpu_mem_util \
    dtdd.teacher.max_num_seqs=$teacher_max_num_seqs \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$agent_loop_cfg \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$teacher_kl_coef \
    +algorithm.filter_groups.enable=$enable_filter_groups \
    +algorithm.filter_groups.metric=$filter_groups_metric \
    +algorithm.filter_groups.reward_std_threshold=$filter_std_threshold \
    +algorithm.token_kl_reg.enable=$token_kl_reg_enable \
    +algorithm.token_kl_reg.gamma=$token_kl_gamma \
    +algorithm.token_kl_reg.beta_min=$token_kl_beta_min \
    +algorithm.token_kl_reg.beta_max=$token_kl_beta_max \
    +algorithm.token_kl_reg.stepwise_enable=$stepwise_enable \
    +algorithm.token_kl_reg.stepwise_epsilon=$stepwise_epsilon \
    +algorithm.token_kl_reg.stepwise_delta=$stepwise_delta \
    +algorithm.token_kl_reg.stepwise_opd_coef=$stepwise_opd_coef \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.custom_cls.path=recipe/demystify/reward.py \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path=recipe/demystify/reward.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$student_model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.loss_agg_mode=$loss_agg_mode \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_len \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    +actor_rollout_ref.ref.model.path=$teacher_model_path \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$logprob_max_len \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=$student_gpu_mem_util \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    reward_model.reward_manager=$reward_manager \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=$ol_enable \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=$ol_buffer_len \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=$ol_pen \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
    +reward_model.reward_kwargs.max_resp_len=$max_response_length \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.log_val_generations=20 \
    trainer.save_freq=${SAVE_FREQ:-10} \
    trainer.test_freq=${TEST_FREQ:-10} \
    trainer.default_local_dir=$default_local_dir \
    trainer.resume_mode=auto \
    trainer.total_epochs=${TOTAL_EPOCHS:-2} "$@"
