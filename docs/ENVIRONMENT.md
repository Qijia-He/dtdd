# Environment (locked)

Every math-lane number in the paper was produced inside one container image,
used unmodified. This page records exactly what that image is and how to
recreate the stack; [`requirements-lock.txt`](../requirements-lock.txt) is the
full `pip freeze` from inside it.

## The image

| | |
|---|---|
| Docker source | `docker.io/verlai/verl:vllm011.latest` (the verl project's official image) |
| Digest at pull time | `sha256:3ce56ff018516b28ab9c4f4fc09d3aa67589074495ace75e2674b720aa4d0e5d` |
| Pulled / built as SIF | 2026-07-21, Apptainer 1.5.2 (image ~14 GB) |
| Base OS / Python | Ubuntu 22.04 / Python 3.12.11 |
| Key pins | torch 2.8.0+cu128 · vLLM 0.11.0 · flash-attn 2.8.1 · ray 2.49.2 · transformers 4.57.1 · tensordict 0.10.0 · hydra-core 1.3.2 |

Rebuild the SIF (digest-pinned, immune to `:latest` moving):

```bash
apptainer build verl_vllm011.sif \
  docker://verlai/verl@sha256:3ce56ff018516b28ab9c4f4fc09d3aa67589074495ace75e2674b720aa4d0e5d
```

Then point the SLURM scripts at it: `export SIF=/path/to/verl_vllm011.sif`.

## What rides on top (nothing is installed into the image)

The image is executed read-only; all project code arrives via `PYTHONPATH`
(see `scripts/train/run_dtdd.sbatch`):

1. `${OPD_ROOT}/verl` — the thunlp/OPD verl fork, fetched by `./setup.sh`
   (symlink into `third_party/OPD/`, pinned at commit `4532fd3`).
2. `${OPD_ROOT}/recipe_dtdd` — the DTDD overlay (this repo).
3. `${EXTRA_PKGS}` — a single pip `--target` directory holding the grader
   dependency the image lacks:

```bash
python3 -m pip install --target="$EXTRA_PKGS" math-verify==0.9.0
# resolves to: math-verify 0.9.0, latex2sympy2_extended 1.11.0,
#              sympy 1.14.0, mpmath 1.3.0
export EXTRA_PKGS=/path/to/that/dir
```

No `pip install -e verl` and no conda env inside the container — if you keep
this shape, "environment drift" is limited to the three paths above.

## Why a container at all

The prebuilt flash-attn 2.8.1 wheel requires a newer glibc than RHEL8's 2.28,
so on RHEL8-family clusters the pip/conda route from upstream thunlp/OPD dies
at import time unless you compile flash-attn yourself. The verl image carries
the exact same stack the upstream conda recipe targets, prebuilt against a
compatible glibc.

## Hand-built (conda) alternative

Follow upstream thunlp/OPD's install instructions
([`docs/UPSTREAM_README.md`](UPSTREAM_README.md)), then diff your env against
`requirements-lock.txt`. The pins that actually matter for reproducing our
runs, in rough order of risk: `vllm==0.11.0`, `torch==2.8.0` (cu128),
`flash_attn==2.8.1`, `transformers==4.57.1`, `ray==2.49.2`,
`tensordict==0.10.0`. Strip the `apptainer exec` wrappers from the sbatch
scripts and export the same `PYTHONPATH` entries.
