#!/bin/bash
#SBATCH --job-name=bench_agentic
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Full-set held-out benchmark of saved checkpoints for one agentic arm.
#
# For each STEP: merge the LoRA adapter onto the base student (node-local),
# render the arm's config as a benchmark-only run whose student is the merged
# model, run it, and delete the merge. Results land under
# $CHECKPOINT_ROOT/<project>/<arm>_bench_<step>/log/benchmark.log and are
# picked up by tools/harvest.py as arm=<arm>, step=<step>.
#
#   ENV=alfworld ARM=dtdd_lora_alfworld STEPS="200 400" \
#     CONFIG=recipe_dtdd_agentic/configs/alfworld/dtdd_lora.yaml \
#     sbatch scripts/eval/run_bench_agentic.sbatch
#
# For a full-fine-tuning arm there is no adapter to merge: set MERGE=0 and the
# checkpoint's actor directory is used as the student directly. For the
# teacher itself (the headroom denominator) set STEPS=0 and STUDENT_MODEL to
# the teacher path; the config's own auxiliary teacher entry stays as it is,
# because the TCOD workflow calls that engine during rollout even when the
# student is the same model.
set -o pipefail
cd ${OPD_ROOT}
mkdir -p logs

SIF=${SIF:?set SIF to your verl+vllm apptainer image}
VENV=${AGENTIC_VENV:?set AGENTIC_VENV to the venv made by scripts/env/make_agentic_venv.sh}
TCOD=${OPD_ROOT}/third_party/TCOD
ENV=${ENV:?alfworld or webshop}
ARM=${ARM:?arm name, e.g. dtdd_lora_alfworld}
STEPS=${STEPS:?space-separated checkpoint steps, or 0 for a raw model}
CONFIG=${CONFIG:?template under recipe_dtdd_agentic/configs/}
MERGE=${MERGE:-1}

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${OPD_ROOT}/agent_ckpts}
CKROOT=${CKROOT:-$CHECKPOINT_ROOT/DTDD_${ENV}/$ARM}
BASE=${STUDENT_MODEL:-models/Qwen2.5-3B-Instruct}
BUFFER_DIR=${BUFFER_DIR:-${OPD_ROOT}/agent_buffers}
HF_HOME=${HF_HOME:-${OPD_ROOT}/hf_cache}
PROJECT=${PROJECT:-DTDD_${ENV}}
if [ "$ENV" = "alfworld" ]; then
  TEACHER_MODEL=${TEACHER_MODEL:-models/GiGPO-Qwen2.5-7B-Instruct-ALFWorld}
  DATA_DIR=${DATA_DIR:-$TCOD/TCOD_examples/alfworld/alfworld_data}
  export ALFWORLD_DATA=${ALFWORLD_DATA:?set ALFWORLD_DATA (see README, ALFWorld data)}
else
  TEACHER_MODEL=${TEACHER_MODEL:-models/GiGPO-Qwen2.5-7B-Instruct-WebShop}
  DATA_DIR=${DATA_DIR:-${OPD_ROOT}/datasets/webshop_tasks}
  WEBSHOP_ROOT=${OPD_ROOT}/third_party/WebShop
fi

JOB_TMP=/tmp/${USER}_${SLURM_JOB_ID}
mkdir -p $JOB_TMP/{triton,inductor,ray,pip,merged,wandb,dummy_lora}
trap "rm -rf $JOB_TMP 2>/dev/null || true" EXIT
_n_gpus=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | wc -l)
echo "host=$(hostname) env=$ENV arm=$ARM steps=$STEPS $(date)"

export TRINITY_DUMMY_LORA_DIR=$JOB_TMP/dummy_lora
export HF_HUB_CACHE=$HF_HOME/hub HF_DATASETS_CACHE=$HF_HOME/datasets HF_HOME
export WANDB_MODE=offline WANDB_DIR=$JOB_TMP/wandb TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES

render() {  # $1 = student path, $2 = run name -> prints rendered yaml path
  local out=$JOB_TMP/cfg_$2.yaml
  sed -e "s#__RUN_NAME__#$2#g" \
      -e "s#__PROJECT__#${PROJECT}#g" \
      -e "s#__CHECKPOINT_ROOT__#${CHECKPOINT_ROOT}#g" \
      -e "s#__STUDENT_MODEL__#$1#g" \
      -e "s#__TEACHER_MODEL__#$(realpath "$TEACHER_MODEL")#g" \
      -e "s#__DATA_DIR__#${DATA_DIR}#g" \
      -e "s#__BUFFER_DIR__#${BUFFER_DIR}#g" \
      -e "s#__HF_HOME__#${HF_HOME}#g" \
      -e "s#__WEBSHOP_ROOT__#${WEBSHOP_ROOT:-}#g" \
      "$CONFIG" > "$out"
  # A benchmark-only run: evaluate the student as loaded, train nothing.
  sed -i -e 's/^\(\s*\)eval_on_startup:.*/\1eval_on_startup: true/' \
         -e 's/^\(\s*\)bench_on_latest_checkpoint:.*/\1bench_on_latest_checkpoint: false/' \
         -e 's/^\(\s*\)continue_from_checkpoint:.*/\1continue_from_checkpoint: false/' \
         -e 's/^\(\s*\)total_steps: [0-9]*\s*$/\1total_steps: 0/' "$out"
  grep -q "__[A-Z_]*__" "$out" && { echo "unrendered placeholder in $out" >&2; return 1; }
  echo "$out"
}

apptainer exec --nv --bind "${BIND_ROOT:-$OPD_ROOT}" "$SIF" bash -c "
  set -o pipefail
  export PATH=$VENV/bin:\$PATH TMPDIR=$JOB_TMP PIP_CACHE_DIR=$JOB_TMP/pip
  export TRITON_CACHE_DIR=$JOB_TMP/triton TORCHINDUCTOR_CACHE_DIR=$JOB_TMP/inductor
  export LD_LIBRARY_PATH=\$(ls -d $VENV/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')\$LD_LIBRARY_PATH
  ${WEBSHOP_ROOT:+export PYTHONPATH=$WEBSHOP_ROOT:\$PYTHONPATH}
  cd $TCOD
  $VENV/bin/python -c 'import torch, vllm, trinity, peft; print(\"env OK\")' || { echo ENV_BROKEN; exit 1; }
  $(declare -f render)
  for STEP in $STEPS; do
    if [ \"\$STEP\" = 0 ]; then
      M=\$(realpath \"$BASE\")
    elif [ \"$MERGE\" = 1 ]; then
      M=$JOB_TMP/merged/\$STEP
      echo \"===== [\$STEP] MERGE \$(date) =====\"
      $VENV/bin/python ${OPD_ROOT}/recipe_dtdd_agentic/scripts/merge_lora.py \
        \"$BASE\" \"$CKROOT/global_step_\$STEP/actor/lora_adapter\" \"\$M\" 2>&1 | tail -3 \
        || { echo \"MERGE_FAIL \$STEP\"; continue; }
    else
      M=$CKROOT/global_step_\$STEP/actor
    fi
    NAME=${ARM}_bench_\$STEP
    CFG=\$(render \"\$M\" \"\$NAME\") || continue
    echo \"===== [\$STEP] BENCH \$(date) =====\"
    $VENV/bin/ray stop --force 2>/dev/null; sleep 2
    $VENV/bin/ray start --head --num-gpus=$_n_gpus --temp-dir=$JOB_TMP/ray 2>&1 | tail -1
    $VENV/bin/trinity run --config \"\$CFG\"; RC=\$?
    echo \"===== [\$STEP] trinity rc=\$RC =====\"
    $VENV/bin/ray stop --force 2>/dev/null
    # Existence is not the same as a result: require the success metric.
    B=\$(ls $CHECKPOINT_ROOT/*/\$NAME/log/benchmark.log 2>/dev/null | head -1)
    if [ -z \"\$B\" ] || ! grep -q 'env_done/mean@1\|success/mean@1' \"\$B\"; then
      echo \"BENCH_EMPTY \$STEP: no success metric in \${B:-<no benchmark.log>}\"
    fi
    [ \"$MERGE\" = 1 ] && [ \"\$STEP\" != 0 ] && rm -rf \"\$M\"
    rm -f \"\$CFG\"
  done
"
echo "BENCH DONE $(date)"
