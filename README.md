# DTDD — Divergence-Triggered Dynamic Distillation

Adaptive teacher takeover for on-policy distillation of math-reasoning LLMs.

This repository contains the math-domain implementation of **DTDD** and the
cluster harness used to run it against the **plain OPD baseline**.

DTDD is an *overlay*, not a fork: everything here is additive, and the OPD/verl
base it attaches to ([thunlp/OPD](https://github.com/thunlp/OPD), Apache-2.0 —
a verl fork in which plain OPD itself lives) is **not** copied into this repo.
`./setup.sh` clones that base at the pinned commit `4532fd3` that every number
in the paper was produced with, and links it in. So what you see below is
exactly the code we wrote: ~500 lines of DTDD plus its launchers, evaluation
and tooling. Upstream's SFT stage (LlamaFactory) is out of scope;
`docs/UPSTREAM_README.md` keeps their original instructions.

## Method

On-policy distillation (OPD) trains a student on its own rollouts against a
frozen teacher's per-token signal. DTDD adds a **divergence tripwire**: while
the student generates, its per-window divergence from the teacher

```
d_k = mean over a 256-token window of | log p_student(t) − log p_teacher(t) |
```

is monitored, and when `d_k > τ` the **teacher takes over** and writes the
continuation. The student then learns from its own prefix via the normal OPD
token-level signal plus the teacher continuation via imitation. Takeover
modes:

| mode | teacher continues from | use |
|---|---|---|
| `rewind` (default) | start of the triggering window | standard tripwire |
| `inherit` | end of the triggering window | keeps the student's window |
| `restart` | the bare question (`cut = 0`) | full teacher demonstration ablation |

Additional machinery: `ε`-random takeovers (exploration control),
constant-rate mode (`DTDD_RATE`: per-batch quantile-τ pinning the takeover
fraction), a minimum-prefix guard, and a teacher sampling **vocab cap** for
same-family teachers whose embedding exceeds the student vocabulary.

Two empirical notes worth knowing before extending this code (details in the
paper): the tripwire **self-extinguishes** as the student improves and re-fires
under destabilization — takeover rate is the primary health telemetry; and
**train-batch accuracy is teacher-inflated** for DTDD (teacher-written tokens
sit in the batch), so all claims must come from held-out evaluation.

## Layout

```
setup.sh                fetches the OPD/verl base at the pinned commit, links it in
recipe_dtdd/            DTDD core — the method, ~500 lines
├── main_dtdd.py        entry point (hydra; swaps the trainer inside Ray's TaskRunner)
├── dtdd_trainer.py     trainer wrapper: d_k scoring, trigger, batch splice, metrics
├── dtdd_takeover.py    d_k computation + splice_batch (origin_mask bookkeeping)
├── teacher_pool.py     frozen-teacher vLLM engine pool (+ sample_vocab_cap)
└── run_dtdd.sh         hydra command builder (env-var driven, see knobs below)
scripts/
├── train/
│   ├── run_dtdd.sbatch     SLURM + apptainer launcher (DTDD arm)
│   ├── run_opd.sbatch      OPD baseline arm
│   ├── run_grpo.sbatch     GRPO (pure RL) reference arm
│   ├── on_policy_distillation_uw.sh  OPD runner (cluster port of upstream's)
│   ├── grpo_uw.sh          GRPO runner (same)
│   └── settings/           one submit script per paper setting (see docs/SETTINGS.md)
├── eval/
│   ├── gen_vllm.py         decoupled generation (vLLM, avg@16, 16384 tokens)
│   ├── grade.py            grading → grading_results.json
│   └── run_eval.sbatch     merge FSDP ckpt → generate → grade, per checkpoint
└── data/
    ├── tag_difficulty.py   difficulty tagging for DAPO-Math-17k
    └── tag_hard_pair.py    builds the hard-gap subset {student pass@4=0 ∩ teacher ≥ .5}
tools/                  optional SLURM automation (auto-eval on checkpoint save,
                        results harvesting, monitor daemon) — see tools/README.md
docs/SETTINGS.md        experiment matrix: models, data, τ, lengths, steps
docs/ENVIRONMENT.md     the locked container environment behind every result
requirements-lock.txt   full pip freeze of that environment

# created by setup.sh, git-ignored:
third_party/OPD/        the thunlp/OPD base @4532fd3 (verl fork, upstream runners,
                        usage examples, benchmark parquets)
verl -> third_party/OPD/verl
datasets/test_data -> third_party/OPD/scripts/val/data
```

## Installation

1. Clone, then fetch the OPD/verl base it overlays onto:

```bash
git clone https://github.com/Qijia-He/opd.git
cd opd && export OPD_ROOT=$(pwd)
./setup.sh          # clones thunlp/OPD@4532fd3 -> third_party/, links verl/
```

`setup.sh` is idempotent and verifies both the pinned commit and the two
upstream symbols the overlay patches, so a moved base fails loudly instead of
silently training something else.

2. Environment — **locked**, see [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).
   Every result was produced inside the official verl container image
   (`verlai/verl:vllm011`, digest pinned there; torch 2.8.0+cu128 / vLLM 0.11.0
   / flash-attn 2.8.1) plus one `pip --target` dir with `math-verify==0.9.0`;
   [`requirements-lock.txt`](requirements-lock.txt) is the full freeze. The
   SLURM scripts assume that image as an apptainer SIF; set `SIF` to its path
   (or strip the `apptainer exec` wrappers and run in a conda env verified
   against the lock file). Placeholders you must set: `OPD_ROOT`, `SIF`,
   `YOUR_ACCOUNT`, `YOUR_GPU_PARTITION`, and optionally `WANDB_ENTITY`.

3. Models and training data (not shipped): student/teacher HF checkpoints under
   `${OPD_ROOT}/models/`, the DAPO-Math-17k parquet under
   `${OPD_ROOT}/datasets/` (`TRAIN_DATASET` in `recipe_dtdd/run_dtdd.sh`). The
   held-out benchmark parquets (AIME24/25, AMC23, MATH-500, …) need no action —
   they come with the base and `setup.sh` links them to `datasets/test_data`.

## Quickstart — one DTDD arm

```bash
cd $OPD_ROOT
DTDD_ENABLE=1 DTDD_TAU=0.5 \
ACTOR_MODEL_PATH=models/DeepSeek-R1-Distill-Qwen-1.5B \
REWARD_MODEL_PATH=models/JustRL-DeepSeek-1.5B \
MAX_RESP_LENGTH=15360 MAX_VAL_RESP_LENGTH=7168 \
sbatch scripts/train/run_dtdd.sbatch
```

The OPD baseline is the same command against `scripts/train/run_opd.sbatch`
(no DTDD variables). Paired runs use `data.shuffle=False` so both arms consume
the same task stream. Full per-setting launch commands:
`scripts/train/settings/` and `docs/SETTINGS.md`.

> **Gotcha:** DTDD is OFF by default in the recipe. An arm launched without
> `DTDD_ENABLE=1` silently trains as plain OPD — verify the teacher-pool
> banner in the log before trusting a run.

## DTDD knobs (environment variables)

| variable | default | meaning |
|---|---|---|
| `DTDD_ENABLE` | `0` | master switch |
| `DTDD_TAU` | setting-specific (0.2–0.7) | fixed divergence threshold |
| `DTDD_RATE` | unset | constant-rate mode: per-batch quantile τ pinning takeover fraction |
| `DTDD_MODE` | `rewind` | `rewind` / `inherit` / `restart` |
| `DTDD_EPS_RANDOM` | `0.0` | probability of a random (untriggered) takeover |
| `DTDD_MIN_PREFIX` | `3` | minimum student turns before takeover is allowed |
| `DTDD_RANDOM_MAX` / `DTDD_MIN_BUDGET` | — | random-trigger window / minimum remaining budget |
| `DTDD_TEACHER_ENGINES` / `DTDD_TEACHER_GPU_UTIL` / `DTDD_TEACHER_GPU_OFFSET` | 1 / 0.12 / — | teacher vLLM pool shape (colocated with the student engine) |
| `DTDD_TEACHER_TEMP` / `DTDD_TEACHER_TOP_P` | 1.0 / 1.0 | teacher sampling |

Training-side knobs (`LR`, `MINI_BATCH_SIZE`, `MAX_RESP_LENGTH`,
`LOG_PROB_TOP_K`, `TOP_K_STRATEGY`, `REWARD_WEIGHT_MODE`, …) are documented at
the top of `recipe_dtdd/run_dtdd.sh`.

> **vLLM memory note:** `gpu_memory_utilization` is a *whole-GPU in-use
> ceiling*, not a per-engine share. When the teacher engine initializes after
> the student engine, its utilization value must cover everything already
> resident on the GPU.

## Evaluation

Held-out evaluation is fully decoupled from training (unbiased sampling,
full-length generation):

```bash
# merge FSDP shards → HF, generate avg@16 on AIME24/AIME25/AMC23, grade
STEPS="20 40 60" sbatch scripts/eval/run_eval.sbatch
```

`grade.py` writes one `grading_results.json` per (checkpoint × benchmark);
aggregate however you like — every number in the paper is a mean over these
files, never a train-batch metric.

## Acknowledgements

Built on [thunlp/OPD](https://github.com/thunlp/OPD) and
[verl](https://github.com/volcengine/verl); teacher/student checkpoints from
their respective releases (DeepSeek-R1-Distill, JustRL, Qwen3). Licensed under
Apache-2.0 (matching upstream).
