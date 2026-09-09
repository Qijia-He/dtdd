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

**4. Agentic lane (optional).** ALFWorld and WebShop run on a different base
(TCOD's trinity runner) and a venv layered over the same container image:

```bash
./setup.sh --agentic                      # + kokolerk/TCOD@465eef4 and princeton-nlp/WebShop@64fa2a5,
                                          #   patched; installs the DTDD workflows
scripts/env/make_agentic_venv.sh /path/to/agentic_venv   # once, from a compute node
export AGENTIC_VENV=/path/to/agentic_venv
```

Then the environment data — see [ALFWorld data](#alfworld-data) and
[WebShop data](#webshop-data). Student and teacher checkpoints go under
`models/` like the math lane's (`Qwen2.5-3B-Instruct`, and the GiGPO-7B
teachers `GiGPO-Qwen2.5-7B-Instruct-ALFWorld` / `-WebShop` from their release).

**5. Code lane (optional).** LiveCodeBench runs on a third base
(`YoungZ365/SOD`, a verl fork) and needs its own Python environment: the SOD
base pins `vllm<=0.9.1`, older than the container image's stack, so the two
cannot share an interpreter.

```bash
./setup.sh --code                        # + YoungZ365/SOD@110c4b8, patched
scripts/env/make_code_env.sh /path/to/code_env   # once; needs conda
export CODE_ENV=/path/to/code_env
```

Then the datasets — see [Code data](#code-data) — and a code-execution
sandbox, see [Sandbox](#sandbox). Student and teacher checkpoints go under
`models/` like the other lanes' (`Qwen3-1.7B-SFT`, and `SOD-GRPO_teacher-4B`
from the SOD release).

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

**Agentic.** One arm per job; the DTDD/OPD pair for a setting differ only in
workflow type and `tau` (see [docs/SETTINGS.md](docs/SETTINGS.md)):

```bash
ENV=alfworld CONFIG=recipe_dtdd_agentic/configs/alfworld/dtdd_lora.yaml \
  sbatch scripts/train/run_dtdd_agentic.sbatch
ENV=alfworld CONFIG=recipe_dtdd_agentic/configs/alfworld/opd_lora.yaml \
  sbatch scripts/train/run_dtdd_agentic.sbatch          # the matched baseline

# full-set held-out benchmark of saved checkpoints (merges the LoRA on the fly)
ENV=alfworld ARM=dtdd_lora_alfworld STEPS="200 400" \
  CONFIG=recipe_dtdd_agentic/configs/alfworld/dtdd_lora.yaml \
  sbatch scripts/eval/run_bench_agentic.sbatch
```

`tools/harvest.py` reads the resulting `benchmark.log` files into the same
results store as the math lane.

**Code.** One arm per job; the DTDD and OPD arms are the same binary with a
different trigger threshold (see [docs/SETTINGS.md](docs/SETTINGS.md)):

```bash
ARM=dtdd sbatch scripts/train/run_dtdd_code.sbatch      # tau = 0.50
ARM=opd  sbatch scripts/train/run_dtdd_code.sbatch      # the matched baseline

# score saved checkpoints on the held-out and LiveCodeBench sets
ARM=qwen3_1p7b_code_dtdd STEPS="60 90 120 131" \
  sbatch scripts/eval/run_eval_code.sbatch
```

Both write into the same `results/eval_results.jsonl` the other lanes use,
so `tools/harvest.py` and `tools/track.py` need no changes.

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
recipe_dtdd_agentic/     ── the method, agentic lanes ──
  workflows/               dtdd_base_workflow.py holds the takeover logic; the
                           ALFWorld and WebShop subclasses supply prompts, the action
                           menu and episode bookkeeping (installed into
                           third_party/TCOD by setup.sh)
  overlay/                 tcod_overlay.patch + PATCH_NOTES.md (harness fixes, not the
                           method); webshop_catalog.patch (120k catalog defaults)
  configs/                 the eight paper arms as __TOKEN__ templates, rendered by the
                           launchers; DTDD/OPD pairs differ only in workflow type and tau
  scripts/                 build_webshop_catalog.py, build_webshop_index.py,
                           make_webshop_tasks.py, merge_lora.py
  data/                    webshop_test_ids.json (the 2,000 held-out goals)
  tests/                   test_smoke.py: import, registry, config validation
scripts/train/run_dtdd_agentic.sbatch    scripts/eval/run_bench_agentic.sbatch
scripts/env/make_agentic_venv.sh         requirements-lock-agentic.txt
recipe_dtdd_code/        ── the method, code lane ──
  recipe/dtdd/             DTDD for the SOD base: agent loop (trigger and
                           takeover), trainer (teacher pool, loss masks),
                           actor (prefix PPO + suffix SFT), teacher server,
                           entry point, and run_dtdd_code.sh (the paper's
                           override bundle). Copied into third_party/SOD by
                           setup.sh
  overlay/                 sod_overlay.patch + PATCH_NOTES.md (an evaluation
                           fix, a metrics fix, four robustness fixes -- not
                           the method); sandbox_runner_stdin.patch for the
                           SandboxFusion image
  data/                    build_code_slices.py, build_lcb_fixed.py, and
                           heldout200_indices.json (the exact held-out split)
  tests/                   test_smoke.py: imports, registry, config, data
scripts/train/run_dtdd_code.sbatch      scripts/eval/run_eval_code.sbatch
scripts/sandbox/                        scripts/env/make_code_env.sh
requirements-lock-code.txt
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

### Agentic lane

Same image, plus a venv. Unlike the math lane, the agentic lane **does**
install into its environment: TCOD needs a newer torch, vLLM and Ray than the
image carries, and ALFWorld, TextWorld and pyserini which it lacks. Those
live in a venv created with the image's Python and put first on `PATH` by the
agentic launchers; the image is still used unmodified underneath.

| | |
|---|---|
| Image | the same `verlai/verl:vllm011` build as above |
| Venv | `scripts/env/make_agentic_venv.sh`; freeze in [`requirements-lock-agentic.txt`](requirements-lock-agentic.txt) |
| Overrides | torch 2.8.0 → **2.9.1** · vLLM 0.11.0 → **0.14.1** · Ray 2.49.2 → **2.56.1** · flash-attn 2.8.1 rebuilt for torch 2.9 · numpy 1.26.4 · scipy 1.13.1 |
| Added | alfworld 0.4.2 · textworld 1.7.0 · pyserini 0.17.0 · peft 0.17.1 |
| Editable | `trinity_rft` → `third_party/TCOD` (the patched checkout is what gets imported) |

The pins that matter, in order of risk: `vllm==0.14.1` (TCOD's supported
ceiling; 0.12.x regresses badly), `torch==2.9.1`, `flash_attn==2.8.1` built
against it, `ray==2.56.1`. Importing torch from this venv needs its bundled
NVIDIA libraries on `LD_LIBRARY_PATH`; the launchers set that.

**Without a container**, follow upstream's install
(`third_party/OPD/README.md` after `setup.sh`) and diff your environment against
[`requirements-lock.txt`](requirements-lock.txt) — the full freeze of the image.
The pins that actually matter, in rough order of risk: `vllm==0.11.0`,
`torch==2.8.0+cu128`, `flash_attn==2.8.1`, `transformers==4.57.1`,
`ray==2.49.2`, `tensordict==0.10.0`. Then strip the `apptainer exec` wrappers
from the sbatch scripts.

### Code lane

Its own environment, not the container. The SOD base pins `vllm<=0.9.1` and
`tensordict<=0.9.1`; the image carries vLLM 0.11 and torch 2.8, so they
cannot share an interpreter. `scripts/env/make_code_env.sh` builds a conda
environment from the pins below and installs the patched SOD checkout into
it as an editable package.

| | |
|---|---|
| Interpreter | conda, Python 3.11 |
| Key pins | torch 2.6.0+cu124 · vLLM 0.8.5.post1 · ray 2.47.1 · transformers 5.12.1 · flash-attn 2.7.4.post1 · numpy 1.26.4 (the base requires numpy<2) |
| Full freeze | [`requirements-lock-code.txt`](requirements-lock-code.txt) |
| Editable | `verl` → `third_party/SOD` (the patched checkout is what gets imported) |
| Also needed | a SandboxFusion server image, see [Sandbox](#sandbox) |

Two wheels are not on PyPI for this torch and CUDA combination —
`flash_attn 2.7.4.post1` and the optional `flashinfer-python 0.2.2.post1` —
and the script fetches them from the projects' release pages. Keep the tags
exactly: a wheel built against a different torch imports fine and then
segfaults.

---

## ALFWorld data

The task lists ship with the TCOD base; the game files do not. Download them
once with the `alfworld` package (in the venv) and point the task lists at
them:

```bash
export ALFWORLD_DATA=/path/to/alfworld_data
$AGENTIC_VENV/bin/alfworld-download          # -> $ALFWORLD_DATA/json_2.1.1, logic/, detectors/
./setup.sh --agentic                          # rewrites the task lists' game-file prefix
```

`setup.sh` regenerates the lists from the pristine upstream copies each time,
so changing `ALFWORLD_DATA` and re-running is safe. The launchers export
`ALFWORLD_DATA` into the job; `alfworld` reads it at import.

## WebShop data

WebShop ships a 1,000-product demo. The experiments use a 120,000-product
catalog chosen to keep **every** goal-bearing product (see the docstring of
`build_webshop_catalog.py` for why a random slice would keep almost none),
a search index over the **full** 1.18M-product release, and a fixed
10,251 / 2,000 goal split. Three steps, all in the venv:

```bash
W=$OPD_ROOT/third_party/WebShop
SNAP=$($AGENTIC_VENV/bin/python -c "from huggingface_hub import snapshot_download as d; \
       print(d('YWZBrandon/webshop-data', repo_type='dataset'))")

# 1. the 120k catalog (seed 42; overlay patch points WebShop's defaults at these files)
$AGENTIC_VENV/bin/python recipe_dtdd_agentic/scripts/build_webshop_catalog.py \
    --items $SNAP/items_shuffle.json --human-instructions $W/data/items_human_ins.json \
    --attributes $SNAP/items_ins_v2.json --output-dir $W/data --catalog-size 120000

# 2. the search index over the full release (pyserini; needs Java 17 on PATH)
cd $W/search_engine
$AGENTIC_VENV/bin/python $OPD_ROOT/recipe_dtdd_agentic/scripts/build_webshop_index.py \
    --snapshot-dir $SNAP --webshop-root $W --output-dir resources_full
$AGENTIC_VENV/bin/python -m pyserini.index.lucene --collection JsonCollection \
    --input resources_full --index indexes_full --generator DefaultLuceneDocumentGenerator \
    --threads 8 --storePositions --storeDocvectors --storeRaw
cd $OPD_ROOT

# 3. the task lists the configs read
$AGENTIC_VENV/bin/python recipe_dtdd_agentic/scripts/make_webshop_tasks.py \
    --output-dir datasets/webshop_tasks
```

Step 2 refuses to run if fewer than a million products load — the symptom of
the paths still resolving to a capped subset. The held-out ids are shipped
verbatim (`recipe_dtdd_agentic/data/webshop_test_ids.json`) rather than as a
seed, so the split is exact.

## Code data

Three builds, all from public sources:

```bash
# 1. the training slice and the 200-problem held-out split
$CODE_ENV/bin/python recipe_dtdd_code/data/build_code_slices.py \
    --source <Open-AgentRL-30K.parquet> \
    --indices recipe_dtdd_code/data/heldout200_indices.json \
    --output-dir datasets/code

# 2. the LiveCodeBench evaluation files, one release at a time
for spec in "test.jsonl v1" "test6.jsonl v6"; do
  set -- $spec
  $CODE_ENV/bin/python recipe_dtdd_code/data/build_lcb_fixed.py \
      --release "$1" --data-source "LiveCodeBench_$2_fixed" \
      --output "datasets/code/lcb_$2_fixed.parquet"
done
```

The training slice keeps every `train-code-*` row of Open-AgentRL-30K (4,205
problems) and subsamples each one's grading tests to at most 12. Code-only
training sends every response to the sandbox, a cost the 30K mix hid because
89% of its rows have a string-match reward; capping the tests in the data
leaves the reward code, which evaluation shares, at full fidelity. The
held-out split ships as the exact 200 row indices rather than a seed.

The LiveCodeBench files are rebuilt for a reason: the parquet in
`Open-AgentRL-Eval` carries one test case per problem and uses it both as
the worked example in the prompt and as the grading truth, so a program that
echoes the shown output scores 1.0. The rebuilt files keep `public_tests`
(prompt only) and `reward_model` (grading, private cases first) apart. The
raw releases are downloaded from `livecodebench/code_generation_lite`; some
are over a gigabyte and a few problems unpickle into gigabytes, so run this
on a machine with real memory, one release at a time.

## Sandbox

Code rewards come from actually running the model's programs, which needs a
[SandboxFusion](https://github.com/bytedance/SandboxFusion) server. Pull the
image once, prepare the patched runner, and the launchers do the rest:

```bash
apptainer pull sandbox-fusion.sif \
  docker://volcengine/sandbox-fusion:server-20250609
export SANDBOX_SIF=$PWD/sandbox-fusion.sif

# the image's own runner, with the stdin guard applied
apptainer exec "$SANDBOX_SIF" cat /root/sandbox/sandbox/runners/base.py \
  > sandbox_base_patched.py
patch sandbox_base_patched.py \
  recipe_dtdd_code/overlay/sandbox_runner_stdin.patch
```

`scripts/sandbox/acquire_sandbox.sh`, sourced by both code-lane launchers,
starts one instance per job and cancels it on exit. Set `SANDBOX_FUSION_URL`
to reuse an existing endpoint instead; it is health-checked before use.

The patch matters: the published image crashes when it kills a timed-out
process, and afterwards answers health checks while every execution fails.
`recipe_dtdd_code/overlay/PATCH_NOTES.md` has the details.

## Acknowledgements & license

Built on [thunlp/OPD](https://github.com/thunlp/OPD) and
[verl](https://github.com/volcengine/verl). Student and teacher checkpoints come
from their respective releases (DeepSeek-R1-Distill, JustRL, Qwen3). Released
under Apache-2.0, matching upstream.
