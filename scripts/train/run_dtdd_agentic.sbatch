#!/bin/bash
#SBATCH --job-name=dtdd_agentic
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Train one agentic arm (ALFWorld or WebShop) with TCOD/trinity inside the
# same verl container the math lane uses, plus the agentic venv layered on top
# (scripts/env/make_agentic_venv.sh; see requirements-lock-agentic.txt).
#
# Pick the arm with CONFIG, a template under recipe_dtdd_agentic/configs/.
# The template's __TOKEN__ placeholders are rendered here from the variables
# below, so one YAML serves every site.
#
#   ENV=alfworld CONFIG=recipe_dtdd_agentic/configs/alfworld/dtdd_lora.yaml \
#     sbatch scripts/train/run_dtdd_agentic.sbatch
#
# The four paper arms per environment are dtdd_lora / opd_lora (LoRA),
# dtdd_fullft / opd_fullft (full fine-tuning), dtdd_gap14x / opd_gap14x
# (14x capacity gap, ALFWorld only). Resume is automatic: the configs set
# continue_from_checkpoint: true and the run name is stable.
set -o pipefail
cd ${OPD_ROOT}
mkdir -p logs

SIF=${SIF:?set SIF to your verl+vllm apptainer image}
VENV=${AGENTIC_VENV:?set AGENTIC_VENV to the venv made by scripts/env/make_agentic_venv.sh}
TCOD=${OPD_ROOT}/third_party/TCOD
ENV=${ENV:?set ENV to alfworld or webshop}
CONFIG=${CONFIG:?set CONFIG to a template under recipe_dtdd_agentic/configs/}

# ---- what the template's placeholders render to -------------------------
RUN_NAME=${RUN_NAME:-$(basename "$CONFIG" .yaml)_${ENV}}
PROJECT=${PROJECT:-DTDD_${ENV}}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${OPD_ROOT}/agent_ckpts}
BUFFER_DIR=${BUFFER_DIR:-${OPD_ROOT}/agent_buffers}
HF_HOME=${HF_HOME:-${OPD_ROOT}/hf_cache}
if [ "$ENV" = "alfworld" ]; then
  STUDENT_MODEL=${STUDENT_MODEL:-models/Qwen2.5-3B-Instruct}
  TEACHER_MODEL=${TEACHER_MODEL:-models/GiGPO-Qwen2.5-7B-Instruct-ALFWorld}
  DATA_DIR=${DATA_DIR:-$TCOD/TCOD_examples/alfworld/alfworld_data}
  # The ALFWorld game files; setup.sh rewrote the task lists to point here.
  export ALFWORLD_DATA=${ALFWORLD_DATA:?set ALFWORLD_DATA (see README, ALFWorld data)}
else
  STUDENT_MODEL=${STUDENT_MODEL:-models/Qwen2.5-3B-Instruct}
  TEACHER_MODEL=${TEACHER_MODEL:-models/GiGPO-Qwen2.5-7B-Instruct-WebShop}
  DATA_DIR=${DATA_DIR:-${OPD_ROOT}/datasets/webshop_tasks}
  WEBSHOP_ROOT=${OPD_ROOT}/third_party/WebShop
fi
mkdir -p "$CHECKPOINT_ROOT" "$BUFFER_DIR" "$HF_HOME"

# Node-local, job-scoped caches: keeps per-job churn off a quota-limited shared fs.
JOB_TMP=/tmp/${USER}_${SLURM_JOB_ID}
mkdir -p $JOB_TMP/{triton,inductor,ray,pip,wandb,dummy_lora}
trap "rm -rf $JOB_TMP 2>/dev/null || true" EXIT

# Render the template. Paths are absolute so the YAML is valid from any cwd.
RENDERED=$JOB_TMP/config.yaml
sed -e "s#__RUN_NAME__#${RUN_NAME}#g" \
    -e "s#__PROJECT__#${PROJECT}#g" \
    -e "s#__CHECKPOINT_ROOT__#${CHECKPOINT_ROOT}#g" \
    -e "s#__STUDENT_MODEL__#$(realpath "$STUDENT_MODEL")#g" \
    -e "s#__TEACHER_MODEL__#$(realpath "$TEACHER_MODEL")#g" \
    -e "s#__DATA_DIR__#${DATA_DIR}#g" \
    -e "s#__BUFFER_DIR__#${BUFFER_DIR}#g" \
    -e "s#__HF_HOME__#${HF_HOME}#g" \
    -e "s#__WEBSHOP_ROOT__#${WEBSHOP_ROOT:-}#g" \
    "$CONFIG" > "$RENDERED"
if grep -q "__[A-Z_]*__" "$RENDERED"; then
  echo "ERROR: unrendered placeholder in $CONFIG:" >&2
  grep -n "__[A-Z_]*__" "$RENDERED" >&2
  exit 1
fi

_n_gpus=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | wc -l)
echo "host=$(hostname) GPUs=$CUDA_VISIBLE_DEVICES env=$ENV config=$CONFIG run=$RUN_NAME $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# The dummy LoRA adapter is throwaway state only this node's engine reads;
# keeping it node-local avoids the shared-filesystem SIGBUS the overlay patch
# describes (trinity/utils/lora_utils.py).
export TRINITY_DUMMY_LORA_DIR=$JOB_TMP/dummy_lora
export HF_HUB_CACHE=$HF_HOME/hub HF_DATASETS_CACHE=$HF_HOME/datasets HF_HOME
export WANDB_MODE=${WANDB_MODE:-offline} WANDB_DIR=$JOB_TMP/wandb
export WANDB_PROJECT=${WANDB_PROJECT:-$PROJECT} WANDB_ENTITY="${WANDB_ENTITY:-}"
export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES  # SLURM sets it alongside CVD; the stack refuses both

apptainer exec --nv --bind "${BIND_ROOT:-$OPD_ROOT}" "$SIF" bash -c "
  set -o pipefail
  export PATH=$VENV/bin:\$PATH TMPDIR=$JOB_TMP PIP_CACHE_DIR=$JOB_TMP/pip
  export TRITON_CACHE_DIR=$JOB_TMP/triton TORCHINDUCTOR_CACHE_DIR=$JOB_TMP/inductor
  # The venv's torch needs its bundled NVIDIA libraries on the loader path.
  export LD_LIBRARY_PATH=\$(ls -d $VENV/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')\$LD_LIBRARY_PATH
  ${WEBSHOP_ROOT:+export PYTHONPATH=$WEBSHOP_ROOT:\$PYTHONPATH}
  cd $TCOD
  $VENV/bin/python -c 'import torch, vllm, trinity, verl; print(\"env OK torch\", torch.__version__, \"vllm\", vllm.__version__)' || { echo ENV_BROKEN; exit 1; }
  $VENV/bin/ray stop --force 2>/dev/null
  $VENV/bin/ray start --head --num-gpus=$_n_gpus --temp-dir=$JOB_TMP/ray 2>&1 | tail -2
  echo '=== trinity run START' \$(date) '==='
  $VENV/bin/trinity run --config $RENDERED
  echo '=== trinity run END rc='\$?' '\$(date) '==='
  $VENV/bin/ray stop --force 2>/dev/null
"
echo "DONE $(date)"
