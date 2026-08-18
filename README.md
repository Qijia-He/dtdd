# DTDD — Divergence-Triggered Dynamic Distillation

**Adaptive teacher takeover for on-policy distillation of math-reasoning LLMs.**

When a student drifts too far from its teacher mid-generation, let the teacher
finish the sequence. This repository is the math-domain implementation of that
idea, plus the cluster harness used to run it head-to-head against plain
on-policy distillation (OPD).

---

## Method

OPD trains a student on its own rollouts against a frozen teacher's per-token
signal. DTDD adds a **divergence tripwire**. While the student generates, its
divergence from the teacher is tracked over 256-token windows:

```
per token t:   δ(t) = Σ_j  w_j(t) · | log p_student(j|t) − log p_teacher(j|t) |
                      over the top-k vocabulary entries j,
                      weighted by w = softmax(student top-k log-probs)

per window k:  d_k  = mean of δ(t) over the 256 tokens of that window
```

The first window where `d_k > τ` fires the tripwire: the **teacher takes over**
and writes the rest of the sequence. The student then learns from two sources in
one batch — its own prefix through the normal OPD token-level signal, and the
teacher's continuation through imitation (behaviour cloning).

| takeover mode | teacher continues from | used for |
|---|---|---|
| `rewind` *(default)* | start of the triggering window | the standard tripwire |
| `inherit` | end of the triggering window | keeps the student's own window |
| `restart` | the bare question (`cut = 0`) | full-demonstration ablation |

Around that core: `ε`-random takeovers for exploration control, a constant-rate
mode (`DTDD_RATE` picks τ per batch as a quantile, pinning the takeover
fraction), a minimum-prefix guard, a minimum-remaining-budget filter, and a
teacher sampling **vocab cap** for teacher/student pairs whose embedding sizes
differ.

Implementation is deliberately non-invasive: `DTDDRayPPOTrainer` subclasses the
upstream trainer and wraps two worker-group methods, so with `DTDD_ENABLE=0` the
entry point is bit-identical to stock OPD. That makes the baseline arm the
*same code path*, not a different program.

---

## Install

**1. Clone and fetch the base.**

```bash
git clone https://github.com/Qijia-He/opd.git
cd opd && export OPD_ROOT=$(pwd)
./setup.sh                    # clones thunlp/OPD@4532fd3 → third_party/, links verl/
```

`setup.sh` is idempotent. It verifies the pinned commit *and* probes for the two
upstream symbols the overlay patches (`RayPPOTrainer`, `token_reward_direct`),
so a moved or rewritten base fails loudly instead of silently training something
else.

**2. Environment.** Every result was produced inside one container image, used
unmodified — see [Environment](#environment) for the digest and rebuild command.
Point the launchers at it and at the one extra package it lacks:

```bash
export SIF=/path/to/verl_vllm011.sif                       # apptainer image
export EXTRA_PKGS=/path/to/pkgs                            # pip install --target "$EXTRA_PKGS" math-verify==0.9.0
```

Also set your scheduler placeholders in `scripts/train/*.sbatch`:
`YOUR_ACCOUNT`, `YOUR_GPU_PARTITION`, and optionally `WANDB_ENTITY`.

**3. Models and training data** (not shipped): student/teacher HF checkpoints
under `models/`, and the DAPO-Math-17k parquet under `datasets/`
(`TRAIN_DATASET` in `recipe_dtdd/run_dtdd.sh`). The held-out benchmark parquets
need no action — they ship with the base and `setup.sh` links them into
`datasets/test_data`.

---

## Quickstart

One DTDD arm:

```bash
cd $OPD_ROOT
DTDD_ENABLE=1 DTDD_TAU=0.5 \
ACTOR_MODEL_PATH=models/DeepSeek-R1-Distill-Qwen-1.5B \
REWARD_MODEL_PATH=models/JustRL-DeepSeek-1.5B \
MAX_RESP_LENGTH=15360 MAX_VAL_RESP_LENGTH=7168 \
sbatch scripts/train/run_dtdd.sbatch
```

Its paired OPD baseline is the same command against
`scripts/train/run_opd.sbatch` with no `DTDD_*` variables. Paired arms run with
`data.shuffle=False` so both consume an identical task stream.

Before committing GPU-days to a new teacher/student pair, calibrate τ with a
short dry run — it builds no teacher pool and logs a `would_trigger@τ` sweep:

```bash
DTDD_ENABLE=1 DTDD_DRY_RUN=1 USER_EXTRA_ARGS="trainer.total_training_steps=10" \
sbatch scripts/train/run_dtdd.sbatch
```

Per-setting launch commands for every experiment in the paper live in
`scripts/train/settings/`, indexed by [docs/SETTINGS.md](docs/SETTINGS.md).

---

## Repo layout

```
setup.sh                 fetches + pins the OPD/verl base, links it into place
recipe_dtdd/             ── the method ──
├── main_dtdd.py         entry point; swaps the trainer inside Ray's TaskRunner
├── dtdd_trainer.py      d_k scoring → trigger → teacher call → splice → BC override
├── dtdd_takeover.py     d_k computation, trigger selection, batch splicing
├── teacher_pool.py      frozen-teacher vLLM engine pool (+ sampling vocab cap)
└── run_dtdd.sh          hydra command builder, env-var driven
scripts/
├── train/
│   ├── run_dtdd.sbatch  SLURM + apptainer launcher — DTDD arm
│   ├── run_opd.sbatch   OPD baseline arm
│   ├── run_grpo.sbatch  GRPO (pure RL) reference arm
│   ├── run_opd.sh       the OPD/GRPO hydra command builders these launchers call
│   ├── run_grpo.sh
│   └── settings/        one submit script per paper setting
├── eval/                merge checkpoint → generate (vLLM) → grade
└── data/                difficulty tagging, hard-gap subset construction
tools/                   optional SLURM automation (see Tooling below)
docs/SETTINGS.md         experiment matrix: models, data, τ, lengths, steps
requirements-lock.txt    full pip freeze of the locked environment

# created by setup.sh, git-ignored:
third_party/OPD/         the base @4532fd3 — verl fork, upstream runners and README,
                         usage examples, benchmark parquets
verl               -> third_party/OPD/verl
datasets/test_data -> third_party/OPD/scripts/val/data
```

---

## Configuration

DTDD is configured entirely through environment variables — no hydra schema
changes, which is what keeps the overlay non-invasive.

| variable | default | meaning |
|---|---|---|
| `DTDD_ENABLE` | `0` | master switch (see [Gotchas](#gotchas)) |
| `DTDD_TAU` | `0.7` | fixed divergence threshold; paper settings use 0.2–0.5 |
| `DTDD_RATE` | `0` (off) | constant-rate mode: per-batch quantile τ pinning the takeover fraction |
| `DTDD_MODE` | `rewind` | `rewind` / `inherit` / `restart` |
| `DTDD_DRY_RUN` | `0` | calibration: no teacher pool, no takeover, logs `would_trigger@τ` |
| `DTDD_EPS_RANDOM` | `0.05` | probability of a random (untriggered) takeover |
| `DTDD_RANDOM_MAX` | `16` | window cap for those random triggers |
| `DTDD_MIN_PREFIX` | `3` | minimum student windows before takeover is allowed |
| `DTDD_MIN_BUDGET` | `256` | minimum remaining tokens for a takeover to be worth it |
| `DTDD_CHUNK` | `256` | divergence window size, in tokens |
| `DTDD_BC_COEF` | `1.0` | imitation weight on teacher-written tokens |
| `DTDD_TEACHER_ENGINES` / `_GPU_UTIL` / `_GPU_OFFSET` | `2` / `0.15` / `0` | teacher vLLM pool shape |
| `DTDD_TEACHER_TEMP` / `_TOP_P` | `1.0` / `1.0` | teacher sampling |

Training-side knobs (`LR`, `MINI_BATCH_SIZE`, `MAX_RESP_LENGTH`,
`LOG_PROB_TOP_K`, `TOP_K_STRATEGY`, `REWARD_WEIGHT_MODE`, …) are documented at
the top of `recipe_dtdd/run_dtdd.sh`.

---

## Evaluation

Held-out evaluation is fully decoupled from training: checkpoints are merged to
HF format and re-generated with unbiased sampling at full length, rather than
read off verl's in-training validation.

```bash
ARM=dtdd_15k CKDIR=$OPD_ROOT/checkpoint/<run> STEPS="20 40 60" \
sbatch scripts/eval/run_eval.sbatch
```

Defaults are avg@16 at temperature 0.7 / top-p 0.95, up to 31744 new tokens
(`EVAL_MAX_TOKENS`), on AIME24 / AIME25 / AMC23; the base also ships MATH-500,
Minerva, Olympiad-Bench, BRUMO25, CMIMC25 and HMMT25. `grade.py` writes one
`grading_results.json` per checkpoint, and `tools/harvest.py` folds those into a
single append-only `results/eval_results.jsonl`.

Every number in the paper is a mean over those files — never a train-batch
metric. See [Gotchas](#gotchas) for why that distinction matters here.

---

## Tooling

`tools/` is optional operational scaffolding; nothing in training or evaluation
depends on it.

| | |
|---|---|
| `autoeval.py` | **Checkpoint saved ⇒ eval submitted.** Watches checkpoint dirs and submits `run_eval.sbatch` once a `global_step_N/` is shard-complete and has stopped changing. Idempotent via a ledger, throttled so evaluation never crowds out training. |
| `harvest.py` | Folds every `grading_results.json` into `results/eval_results.jsonl`. |
| `monitor_daemon.sh`, `track.py` | Annotated site-specific references: stall detection, auto-resume, disk/inode sentinels, live dashboard. Adapt before use. |

The reason `autoeval.py` exists is worth stating: every coverage hole in our
experiment record came from *train now, evaluate later* — checkpoints that were
never graded, or that a disk cleanup removed before grading. Evaluating at save
time makes the **result** durable independently of the checkpoint, which is the
difference between a 200 KB results file and terabytes you cannot afford to keep.

---

## Environment

| | |
|---|---|
| Image | `verlai/verl:vllm011` (the verl project's official image), used unmodified |
| Digest | `sha256:3ce56ff018516b28ab9c4f4fc09d3aa67589074495ace75e2674b720aa4d0e5d` |
| Base | Ubuntu 22.04, Python 3.12.11, CUDA 12.8 |
| Key pins | torch 2.8.0+cu128 · vLLM 0.11.0 · flash-attn 2.8.1 · ray 2.49.2 · transformers 4.57.1 · tensordict 0.10.0 · hydra-core 1.3.2 |

```bash
apptainer build verl_vllm011.sif \
  docker://verlai/verl@sha256:3ce56ff018516b28ab9c4f4fc09d3aa67589074495ace75e2674b720aa4d0e5d
```

Nothing is installed *into* the image. All project code arrives on `PYTHONPATH`:
`$OPD_ROOT/verl` (the fetched base), `$OPD_ROOT/recipe_dtdd` (this overlay), and
`$EXTRA_PKGS` — a single `pip --target` directory holding `math-verify==0.9.0`
(with `latex2sympy2_extended` 1.11.0, `sympy` 1.14.0, `mpmath` 1.3.0), the one
grading dependency the image lacks. Keeping this shape means environment drift
is confined to three paths.

**Why a container.** The prebuilt flash-attn 2.8.1 wheel needs a newer glibc
than RHEL8's 2.28, so on RHEL8-family clusters the upstream pip/conda route dies
at import unless you compile flash-attn yourself. This image carries the same
stack the upstream recipe targets, prebuilt against a compatible glibc.

**Without a container**, follow upstream's install
(`third_party/OPD/README.md` after `setup.sh`) and diff your environment against
[`requirements-lock.txt`](requirements-lock.txt) — the full freeze of the image.
The pins that actually matter, in rough order of risk: `vllm==0.11.0`,
`torch==2.8.0+cu128`, `flash_attn==2.8.1`, `transformers==4.57.1`,
`ray==2.49.2`, `tensordict==0.10.0`. Then strip the `apptainer exec` wrappers
from the sbatch scripts.

---

## Acknowledgements & license

Built on [thunlp/OPD](https://github.com/thunlp/OPD) and
[verl](https://github.com/volcengine/verl). Student and teacher checkpoints come
from their respective releases (DeepSeek-R1-Distill, JustRL, Qwen3). Released
under Apache-2.0, matching upstream.
