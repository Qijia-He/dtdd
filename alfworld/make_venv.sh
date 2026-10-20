#!/bin/bash
# Build the ALFWorld virtualenv on top of the verl container image.
#
# The math example installs nothing into the image. ALFWorld cannot do
# that: TCOD needs a newer torch/vLLM/ray than the image carries, plus
# alfworld and textworld, which it lacks entirely. So a venv is
# created with the image's Python 3.12 and layered over it at run time (the
# launchers put $AGENTIC_VENV/bin first on PATH). The resulting pins are in
# alfworld/requirements-lock.txt; this script reproduces them.
#
#   export SIF=/path/to/verl_vllm011.sif
#   export DTDD_ROOT=/path/to/dtdd       # after ./setup.sh --alfworld
#   alfworld/make_venv.sh /path/to/agentic_venv
#   export AGENTIC_VENV=/path/to/agentic_venv
#
# Runs inside the container (apptainer must be able to start it, which on
# most clusters means from a compute node, not a login node). Idempotent:
# re-running verifies and fills in whatever is missing.
set -euo pipefail

VENV=${1:?usage: alfworld/make_venv.sh <venv-dir>}
SIF=${SIF:?set SIF to your verl+vllm apptainer image}
DTDD_ROOT=${DTDD_ROOT:?set DTDD_ROOT (run ./setup.sh --agentic first)}
TCOD=$DTDD_ROOT/third_party/TCOD
[ -d "$TCOD/trinity" ] || { echo "third_party/TCOD missing: run ./setup.sh --agentic" >&2; exit 1; }

# Pins behind every agentic number in the paper (see alfworld/requirements-lock.txt).
VLLM_PIN=0.14.1          # top of TCOD's allowed range; earlier 0.12.x regressed badly
FLASH_ATTN_PIN=2.8.1     # rebuilt against the venv's torch 2.9
NUMPY_PIN=1.26.4
SCIPY_PIN=1.13.1

# Installs are constrained to the frozen versions in alfworld/requirements-lock.txt so the
# venv does not drift as PyPI moves on. Only plain name==version pins are used: the lock
# also lists image builds (git/URL/local wheels) that pip cannot fetch.
CONSTRAINTS="$VENV.constraints.txt"
grep -E '^[A-Za-z0-9_.-]+==' "$DTDD_ROOT/alfworld/requirements-lock.txt" | grep -v '+' |
  grep -v -i -E '^(flash[-_]attn|trinity[-_]rft)==' > "$CONSTRAINTS"

apptainer exec --nv --bind "$DTDD_ROOT" --bind "$(dirname "$VENV")" "$SIF" bash -c "
  set -euo pipefail
  # Build steps call a bare 'python' (e.g. fast-downward-textworld's build.py). The host
  # PATH reaches the container, and a host interpreter there cannot even import the
  # standard library inside it, so pin PATH to the venv and the image.
  export PATH='$VENV/bin':/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  unset PYTHONPATH PYTHONHOME PYTHONSTARTUP
  if [ ! -x '$VENV/bin/python' ]; then
    echo '== creating venv with the image python' \$(python3 --version)
    python3 -m venv '$VENV'
  fi
  P='$VENV/bin/python'
  \$P -m pip install --quiet --upgrade pip
  cd '$TCOD'
  # trinity_rft is an EDITABLE install of the patched checkout, so the overlay
  # and the DTDD workflows are what gets imported.
  \$P -m pip install --quiet -c '$CONSTRAINTS' -e '.[vllm,dev]'
  \$P -m pip install --quiet -c '$CONSTRAINTS' 'vllm==$VLLM_PIN' alfworld
  \$P -m pip install --quiet --no-deps 'numpy==$NUMPY_PIN' 'scipy==$SCIPY_PIN'
  # torch 2.9 needs its bundled NVIDIA libs on the loader path for any import.
  export LD_LIBRARY_PATH=\$(ls -d '$VENV'/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')\${LD_LIBRARY_PATH:-}
  if ! \$P -c 'from flash_attn import flash_attn_func' 2>/dev/null; then
    echo '== compiling flash-attn $FLASH_ATTN_PIN for the venv torch (needs nvcc; several minutes)'
    \$P -m pip install --quiet --no-build-isolation --force-reinstall --no-deps 'flash-attn==$FLASH_ATTN_PIN'
  fi
  \$P - <<'PY'
import importlib
want = {'torch': '2.9.1', 'vllm': '$VLLM_PIN', 'trinity': None, 'verl': '0.7.0',
        'peft': None, 'alfworld': None, 'textworld': None,
        'flash_attn': '$FLASH_ATTN_PIN', 'numpy': '$NUMPY_PIN', 'scipy': '$SCIPY_PIN'}
bad = []
for name, pin in want.items():
    try:
        mod = importlib.import_module(name)
    except Exception as e:  # noqa: BLE001
        bad.append(f'{name}: import failed ({type(e).__name__}: {e})'); continue
    ver = getattr(mod, '__version__', '?')
    if pin and not ver.startswith(pin):
        bad.append(f'{name}: {ver} != {pin}')
    print(f'  {name:<11} {ver}')
if bad:
    print('MISMATCH:'); print('\n'.join('  ' + b for b in bad)); raise SystemExit(1)
print('venv OK')
PY
"
echo "== done. export AGENTIC_VENV=$VENV"
echo "   compare against alfworld/requirements-lock.txt with:"
echo "   apptainer exec $SIF $VENV/bin/python -m pip freeze | diff - $DTDD_ROOT/alfworld/requirements-lock.txt"
