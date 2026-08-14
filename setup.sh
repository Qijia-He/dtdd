#!/bin/bash
# Fetch the OPD/verl base this recipe overlays onto.
#
# DTDD is an overlay: recipe_dtdd/ subclasses the trainer from thunlp/OPD's verl
# fork (which is where plain OPD itself lives -- the token-level distillation
# reward is `token_reward_direct` in verl/verl/trainer/ppo/core_algos.py). That
# base is NOT vendored here; this script clones it at the exact commit every
# number in the paper was produced with, and links it into place.
#
#   ./setup.sh              # clone/verify the base, create the verl symlink
#   ./setup.sh --force      # re-clone from scratch
#
# Idempotent: safe to re-run. Creates third_party/OPD/ and the verl -> symlink,
# both git-ignored.
set -euo pipefail

UPSTREAM_URL=${UPSTREAM_URL:-https://github.com/thunlp/OPD.git}
UPSTREAM_PIN=${UPSTREAM_PIN:-4532fd3}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEST=$ROOT/third_party/OPD

if [ "${1:-}" = "--force" ]; then
    echo "== removing existing $DEST"
    rm -rf "$DEST"
fi

if [ ! -d "$DEST/.git" ]; then
    echo "== cloning $UPSTREAM_URL -> third_party/OPD"
    mkdir -p "$ROOT/third_party"
    git clone --quiet "$UPSTREAM_URL" "$DEST"
fi

echo "== checking out pinned commit $UPSTREAM_PIN"
git -C "$DEST" fetch --quiet origin || true
if ! git -C "$DEST" cat-file -e "${UPSTREAM_PIN}^{commit}" 2>/dev/null; then
    echo "ERROR: commit $UPSTREAM_PIN not found in $UPSTREAM_URL." >&2
    echo "  Upstream may have rewritten history. The DTDD overlay patches" >&2
    echo "  internals of that fork (rm_wg.compute_rm_score," >&2
    echo "  compute_distillation_reward) and is not guaranteed to apply to a" >&2
    echo "  different revision. Override with UPSTREAM_PIN=<sha> at your own risk." >&2
    exit 1
fi
git -C "$DEST" -c advice.detachedHead=false checkout --quiet "$UPSTREAM_PIN"

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

# Sanity: the two symbols the overlay depends on must exist in what we just fetched.
for probe in \
    "verl/verl/trainer/ppo/ray_trainer.py:RayPPOTrainer" \
    "verl/verl/trainer/ppo/core_algos.py:token_reward_direct"; do
    f=${probe%%:*}; sym=${probe##*:}
    if ! grep -q "$sym" "$ROOT/$f" 2>/dev/null; then
        echo "ERROR: expected symbol '$sym' not found in $f -- wrong base revision?" >&2
        exit 1
    fi
done

cat <<EOF

Base ready.
  third_party/OPD   $(git -C "$DEST" rev-parse --short HEAD) ($(git -C "$DEST" log -1 --format=%s))
  verl ->           third_party/OPD/verl
  eval parquets     third_party/OPD/scripts/val/data  (AIME24/25, AMC23, MATH-500, ...)

Next:
  1. export OPD_ROOT=$ROOT
  2. export SIF=/path/to/verl_vllm011.sif        # see docs/ENVIRONMENT.md
     export EXTRA_PKGS=/path/to/math-verify-dir  # pip install --target
  3. put student/teacher HF checkpoints under models/ and the training parquet
     under datasets/ (see README "Models and data")
  4. sbatch scripts/train/run_dtdd.sbatch        # or run_opd / run_grpo
EOF
