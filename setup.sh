#!/bin/bash
# Fetch the upstream bases this repository overlays onto.
#
# DTDD is an overlay in every lane. Nothing upstream is vendored here; this
# script clones each base at the exact commit every number in the paper was
# produced with, applies the small patches the lanes need, and links things
# into place.
#
#   math lane     thunlp/OPD@4532fd3      -> third_party/OPD, linked as verl/
#   agentic lane  kokolerk/TCOD@465eef4   -> third_party/TCOD  (ALFWorld, WebShop)
#                 princeton-nlp/WebShop@64fa2a5 -> third_party/WebShop
#
#   ./setup.sh                 # math lane only (the default)
#   ./setup.sh --agentic       # math + agentic lanes
#   ./setup.sh --force ...     # re-clone from scratch first
#
# Idempotent: safe to re-run. Everything it creates is git-ignored.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FORCE=0; AGENTIC=0
for arg in "$@"; do
    case "$arg" in
        --force)   FORCE=1 ;;
        --agentic) AGENTIC=1 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

# fetch_pinned <url> <sha> <dest> <what-the-overlay-touches>
# Clones (or verifies) a base at a pinned commit. A missing commit is fatal,
# because the overlays patch internals and are not guaranteed to apply to any
# other revision.
fetch_pinned() {
    local url=$1 pin=$2 dest=$3 touches=$4
    if [ "$FORCE" = 1 ]; then
        echo "== removing existing $dest"
        rm -rf "$dest"
    fi
    if [ ! -d "$dest/.git" ]; then
        echo "== cloning $url -> ${dest#$ROOT/}"
        mkdir -p "$(dirname "$dest")"
        git clone --quiet "$url" "$dest"
    fi
    echo "== checking out pinned commit $pin in ${dest#$ROOT/}"
    git -C "$dest" fetch --quiet origin || true
    if ! git -C "$dest" cat-file -e "${pin}^{commit}" 2>/dev/null; then
        echo "ERROR: commit $pin not found in $url." >&2
        echo "  Upstream may have rewritten history. The overlay patches $touches" >&2
        echo "  and is not guaranteed to apply to a different revision." >&2
        exit 1
    fi
    git -C "$dest" -c advice.detachedHead=false checkout --quiet "$pin"
}

# apply_overlay <dest> <patch>
# Applies a patch only if it is not already applied (so re-runs are no-ops),
# and refuses to leave a half-applied tree.
apply_overlay() {
    local dest=$1 patch=$2
    if git -C "$dest" apply --reverse --check "$patch" 2>/dev/null; then
        echo "== overlay already applied: $(basename "$patch")"
        return
    fi
    if ! git -C "$dest" apply --check "$patch"; then
        echo "ERROR: $(basename "$patch") does not apply to ${dest#$ROOT/}." >&2
        echo "  Run with --force to re-clone, or inspect with: git -C $dest status" >&2
        exit 1
    fi
    git -C "$dest" apply "$patch"
    echo "== applied $(basename "$patch")"
}

# probe <file> <symbol>...  -- every symbol must appear in the file.
probe() {
    local f=$1; shift
    for sym in "$@"; do
        if ! grep -q "$sym" "$ROOT/$f" 2>/dev/null; then
            echo "ERROR: expected symbol '$sym' not found in $f -- wrong base revision or overlay not applied?" >&2
            exit 1
        fi
    done
}

# ---------------------------------------------------------------- math lane
fetch_pinned "${UPSTREAM_URL:-https://github.com/thunlp/OPD.git}" \
             "${UPSTREAM_PIN:-4532fd3}" "$ROOT/third_party/OPD" \
             "rm_wg.compute_rm_score and compute_distillation_reward"

# recipe_dtdd/main_dtdd.py resolves the hydra config dir as ../verl/verl/trainer/config,
# and the launchers put ${OPD_ROOT}/verl on PYTHONPATH, so verl must sit at the repo root.
ln -sfn third_party/OPD/verl "$ROOT/verl"

# The training runners read validation parquets from datasets/test_data, and the
# upstream base ships exactly that layout (AIME24/test.parquet, ...). Link it if the
# user has not provided their own.
if [ ! -e "$ROOT/datasets/test_data" ]; then
    mkdir -p "$ROOT/datasets"
    ln -sfn ../third_party/OPD/scripts/val/data "$ROOT/datasets/test_data"
    echo "== linked datasets/test_data -> third_party/OPD/scripts/val/data"
fi

probe verl/verl/trainer/ppo/ray_trainer.py RayPPOTrainer
probe verl/verl/trainer/ppo/core_algos.py token_reward_direct

# ------------------------------------------------------------- agentic lane
if [ "$AGENTIC" = 1 ]; then
    AG=$ROOT/recipe_dtdd_agentic
    TCOD=$ROOT/third_party/TCOD
    WEBSHOP=$ROOT/third_party/WebShop

    fetch_pinned "${TCOD_URL:-https://github.com/kokolerk/TCOD.git}" \
                 "${TCOD_PIN:-465eef4}" "$TCOD" \
                 "trinity/buffer/utils.py, vllm_model.py, explorer.py and lora_utils.py"
    apply_overlay "$TCOD" "$AG/overlay/tcod_overlay.patch"

    # The DTDD workflows are the method; they are installed as new files at
    # the module paths the overlay registers in trinity/common/workflows.
    for env in alfworld webshop; do
        src=$AG/workflows/dtdd_${env}_workflow.py
        dst=$TCOD/trinity/common/workflows/envs/TCOD/$env/DTDD_workflow.py
        if ! cmp -s "$src" "$dst" 2>/dev/null; then
            cp "$src" "$dst"
            echo "== installed workflows/dtdd_${env}_workflow.py -> $(realpath --relative-to="$ROOT" "$dst")"
        fi
    done

    # ALFWorld task lists ship with the base under the original author's
    # absolute game-file paths. Rewrite that prefix to ALFWORLD_DATA. Done from
    # the pristine upstream copy each time, so re-running with a different
    # ALFWORLD_DATA is correct and never stacks rewrites.
    if [ -n "${ALFWORLD_DATA:-}" ]; then
        data=$TCOD/TCOD_examples/alfworld/alfworld_data
        for f in test test_unseen train train_expert train_hard; do
            git -C "$TCOD" show "${TCOD_PIN:-465eef4}:TCOD_examples/alfworld/alfworld_data/$f.jsonl" \
              | sed "s#/nas/wjq/alfworld/#${ALFWORLD_DATA%/}/#g" > "$data/$f.jsonl"
        done
        echo "== ALFWorld task lists point at ALFWORLD_DATA=$ALFWORLD_DATA"
    else
        echo "-- ALFWORLD_DATA not set: ALFWorld task lists still carry upstream's paths;"
        echo "   export ALFWORLD_DATA=/path/to/alfworld/data and re-run (see README)."
    fi

    fetch_pinned "${WEBSHOP_URL:-https://github.com/princeton-nlp/WebShop.git}" \
                 "${WEBSHOP_PIN:-64fa2a5}" "$WEBSHOP" \
                 "web_agent_site/utils.py default catalog paths"
    apply_overlay "$WEBSHOP" "$AG/overlay/webshop_catalog.patch"

    probe third_party/TCOD/trinity/buffer/utils.py "A context manager that yields a session exactly once"
    probe third_party/TCOD/trinity/common/models/vllm_model.py _wait_adapter_complete
    probe third_party/TCOD/trinity/common/workflows/__init__.py DTDD_alfworld_workflow DTDD_webshop_workflow
    probe third_party/WebShop/web_agent_site/utils.py items_shuffle_120000
fi

# ------------------------------------------------------------------ summary
cat <<EOF

Base ready.
  third_party/OPD   $(git -C "$ROOT/third_party/OPD" rev-parse --short HEAD) ($(git -C "$ROOT/third_party/OPD" log -1 --format=%s))
  verl ->           third_party/OPD/verl
  eval parquets     third_party/OPD/scripts/val/data  (AIME24/25, AMC23, MATH-500, ...)
EOF
if [ "$AGENTIC" = 1 ]; then
cat <<EOF
  third_party/TCOD     $(git -C "$ROOT/third_party/TCOD" rev-parse --short HEAD) + tcod_overlay.patch + DTDD workflows
  third_party/WebShop  $(git -C "$ROOT/third_party/WebShop" rev-parse --short HEAD) + webshop_catalog.patch
EOF
fi
cat <<EOF

Next:
  1. export OPD_ROOT=$ROOT
  2. export SIF=/path/to/verl_vllm011.sif        # see README "Environment"
     export EXTRA_PKGS=/path/to/math-verify-dir  # pip install --target
  3. put student/teacher HF checkpoints under models/ and the training parquet
     under datasets/ (see README "Models and data")
  4. sbatch scripts/train/run_dtdd.sbatch        # or run_opd / run_grpo
EOF
if [ "$AGENTIC" = 1 ]; then
cat <<EOF
  agentic lane, additionally:
  5. scripts/env/make_agentic_venv.sh            # once; then export AGENTIC_VENV
  6. ALFWorld: alfworld-download; export ALFWORLD_DATA; re-run ./setup.sh --agentic
     WebShop:  see README "WebShop data" (catalog, search index, task lists)
  7. ENV=alfworld CONFIG=recipe_dtdd_agentic/configs/alfworld/dtdd_lora.yaml \\
       sbatch scripts/train/run_dtdd_agentic.sbatch
EOF
fi
