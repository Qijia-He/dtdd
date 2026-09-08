#!/bin/bash
# Build the code-lane Python environment.
#
#   OPD_ROOT=/path/to/opd scripts/env/make_code_env.sh /path/to/code_env
#   export CODE_ENV=/path/to/code_env
#
# Why this lane gets its own environment, when the math lane installs nothing
# and the agentic lane layers a venv over the container image: the SOD base
# pins vllm<=0.9.1, which is older than the stack in the container image
# (torch 2.8 / vLLM 0.11). The two cannot share one interpreter, so the code
# lane runs in a plain conda environment built from the pins in
# requirements-lock-code.txt.
#
# Needs conda (or mamba) on PATH, a CUDA 12.4-capable driver, and a compiler
# for the two wheels that are not on PyPI for this combination. Re-running is
# safe: an existing environment is reused and only missing pieces installed.
set -euo pipefail

ENV_DIR=${1:?usage: make_code_env.sh <env-dir>}
OPD_ROOT=${OPD_ROOT:?set OPD_ROOT (run ./setup.sh --code first)}
SOD="$OPD_ROOT/third_party/SOD"
[ -d "$SOD/verl" ] || {
  echo "third_party/SOD missing: run ./setup.sh --code" >&2; exit 1; }

PYTHON_PIN=3.11
TORCH_PIN=2.6.0
VLLM_PIN=0.8.5.post1
# Neither wheel is published on PyPI for cu124 + torch 2.6, so both come from
# the projects' own release pages. Keep the tags exactly as written: a wheel
# built against a different torch or ABI imports but segfaults at first use.
FLASH_ATTN_TAG=v2.7.4.post1
FLASH_ATTN_WHEEL=flash_attn-2.7.4.post1
FLASH_ATTN_WHEEL+=+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
FLASH_ATTN_URL=https://github.com/Dao-AILab/flash-attention/releases
FLASH_ATTN_URL+=/download/$FLASH_ATTN_TAG/$FLASH_ATTN_WHEEL
FLASHINFER_TAG=v0.2.2.post1
FLASHINFER_WHEEL=flashinfer_python-0.2.2.post1
FLASHINFER_WHEEL+=+cu124torch2.6-cp38-abi3-linux_x86_64.whl
FLASHINFER_URL=https://github.com/flashinfer-ai/flashinfer/releases
FLASHINFER_URL+=/download/$FLASHINFER_TAG/$FLASHINFER_WHEEL

if [ ! -x "$ENV_DIR/bin/python" ]; then
  echo "== creating conda environment with python $PYTHON_PIN"
  conda create --yes --prefix "$ENV_DIR" "python=$PYTHON_PIN"
fi
PYTHON="$ENV_DIR/bin/python"

echo "== installing torch $TORCH_PIN (cu124) and vLLM $VLLM_PIN"
"$PYTHON" -m pip install --quiet --upgrade pip
"$PYTHON" -m pip install --quiet \
    --index-url https://download.pytorch.org/whl/cu124 \
    "torch==$TORCH_PIN" torchvision torchaudio
"$PYTHON" -m pip install --quiet "vllm==$VLLM_PIN"

echo "== installing the SOD base (editable) and its extras"
"$PYTHON" -m pip install --quiet -e "$SOD[vllm,math]"

if ! "$PYTHON" -c 'import flash_attn' 2>/dev/null; then
  echo "== installing flash-attn (prebuilt wheel)"
  "$PYTHON" -m pip install --quiet --no-deps "$FLASH_ATTN_URL"
fi
if ! "$PYTHON" -c 'import flashinfer' 2>/dev/null; then
  echo "== installing flashinfer (prebuilt wheel)"
  "$PYTHON" -m pip install --quiet --no-deps "$FLASHINFER_URL" || \
    echo "   flashinfer is optional; continuing without it" >&2
fi

echo "== verifying"
"$PYTHON" - <<PY
import importlib

want = {"torch": "$TORCH_PIN", "vllm": "$VLLM_PIN", "verl": None,
        "ray": None, "transformers": None, "flash_attn": None,
        "hydra": None, "omegaconf": None, "tensordict": None,
        "pyarrow": None, "datasets": None}
bad = []
for name, pin in want.items():
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        bad.append(f"{name}: import failed ({type(exc).__name__}: {exc})")
        continue
    version = getattr(module, "__version__", "?")
    if pin and not version.startswith(pin):
        bad.append(f"{name}: {version} != {pin}")
    print(f"  {name:<13} {version}")
if bad:
    print("MISMATCH:")
    print("\n".join("  " + line for line in bad))
    raise SystemExit(1)
print("environment OK")
PY

echo "== done. export CODE_ENV=$ENV_DIR"
echo "   compare against the recorded pins with:"
echo "   $PYTHON -m pip freeze | diff - $OPD_ROOT/requirements-lock-code.txt"
