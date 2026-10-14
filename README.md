<div align="center">
  <h1>DTDD: Divergence-Triggered Dynamic Distillation<br>for Reliable On-Policy Supervision</h1>
</div>
<p align="center">
  <a href="LICENSE">
    <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue">
  </a>
  <a href="https://dtdd-proj.github.io/">
    <img alt="Project page" src="https://img.shields.io/badge/DTDD-project%20page-2f6f52">
  </a>
  <a href="https://dtdd-proj.github.io/">
    <img alt="Paper" src="https://img.shields.io/badge/paper-coming%20soon-lightgrey">
  </a>
  <a href="#citation">
    <img alt="Citation" src="https://img.shields.io/badge/cite-BibTeX-orange">
  </a>
</p>

<p align="center">
  <img src="assets/overview.png" width="92%" alt="DTDD pipeline: the student generates, segment-level divergence is monitored, and the teacher takes over the rest of the sequence once a segment diverges past the threshold.">
</p>

## Overview

On-policy distillation (OPD) supervises a student on its own rollouts with a
frozen teacher's per-token signal. That signal becomes unreliable once the
student drifts into states the teacher would never visit, which happens with a
large initial policy gap and with errors that compound over long generations
and multi-turn tool use.

DTDD monitors student–teacher divergence segment by segment while the student
generates. When a segment's divergence exceeds a threshold τ, the teacher takes
over from that point and writes the rest of the sequence. The student then
learns from both parts of one sequence: its own prefix through the standard OPD
signal, and the teacher's continuation through behaviour cloning.

This repository is a minimal implementation with one runnable example per
setting in the paper, on the same models and data:

| Example | Student ← teacher | Data | Paper |
|---|---|---|---|
| [Math](#math) | Qwen3-1.7B-Base ← Qwen3-4B-Base-GRPO | DAPO-Math-17K → AIME24/25, AMC23 | design-control table |
| [ALFWorld](#alfworld) | Qwen2.5-3B-Instruct (LoRA) ← GiGPO-Qwen2.5-7B | ALFWorld | agentic results |
| [Coding](#coding) | Qwen3-1.7B (SFT) ← SOD-GRPO-teacher-4B | Open-AgentRL-30K (code) → held-out, LiveCodeBench | agentic results |

Each example is an overlay on an existing open-source trainer. `setup.sh`
fetches that trainer at the exact commit the paper used and applies a small
patch; DTDD itself is the handful of files in each directory. With DTDD
switched off, every example runs plain OPD on the same code path, which is how
the baselines were produced.

## Method to code

| Step | Math | ALFWorld | Coding |
|---|---|---|---|
| Per-segment divergence d_k | `math/dtdd_takeover.py: compute_dk_from_diff` | `alfworld/workflows/dtdd_base_workflow.py: _compute_turn_divergence` | `code/recipe/dtdd/dtdd_agent_loop.py: DTDDToolAgentLoop.run` |
| Trigger (first d_k > τ) | `dtdd_takeover.py: pick_triggers` | `dtdd_base_workflow.py: _run_episode` | `dtdd_agent_loop.py: DTDDToolAgentLoop.run` |
| Teacher writes the suffix | `math/teacher_pool.py: TeacherGenPool`, `dtdd_takeover.py: splice_batch` | `dtdd_base_workflow.py: _run_episode`, `_rescore_teacher_turn` | `dtdd_agent_loop.py: DTDDTeacherServerManager` |
| OPD on prefix + BC on suffix | `math/dtdd_trainer.py: DTDDRayPPOTrainer._wrap_workers` | `dtdd_base_workflow.py` (per-turn records) | `code/recipe/dtdd/dtdd_dp_actor.py: update_policy` |

The knobs are environment variables (`DTDD_TAU`, `DTDD_CHUNK`, `DTDD_MODE`, …);
the defaults are listed at the top of `math/dtdd_trainer.py`,
`alfworld/configs/dtdd_lora.yaml` and `code/recipe/dtdd/config/dtdd_trainer.yaml`.

## Setup

All three examples need Linux, NVIDIA GPUs (the paper's runs used 4 per run;
the math example runs on as few as 2: one for the student, one for the teacher;
80 GB A100 or 44–48 GB L40S/A40), [Apptainer](https://apptainer.org/), and SLURM for the
provided launchers. The launchers are short; strip the `#SBATCH` lines and the
`apptainer exec` wrapper to run elsewhere.

```bash
git clone https://github.com/Qijia-He/dtdd.git && cd dtdd
export DTDD_ROOT=$PWD

./setup.sh                      # math only
./setup.sh --alfworld --code    # all three examples

# The container every math/ALFWorld run uses (verl's official image, unmodified).
apptainer build verl_vllm011.sif \
  docker://verlai/verl@sha256:3ce56ff018516b28ab9c4f4fc09d3aa67589074495ace75e2674b720aa4d0e5d
export SIF=$PWD/verl_vllm011.sif

# The one grading dependency the image lacks.
pip install --target=$PWD/extra_pkgs math-verify==0.9.0 && export EXTRA_PKGS=$PWD/extra_pkgs
```

Checkpoints go under `models/` and data under `datasets/` (both git-ignored).
Edit the `--account` / `--partition` lines of each `*.sbatch` for your cluster.

## Math

```bash
huggingface-cli download Qwen/Qwen3-1.7B-Base             --local-dir models/Qwen3-1.7B-Base
huggingface-cli download Thinking-Space/Qwen3-4B-Base-GRPO --local-dir models/Qwen3-4B-Base-GRPO
python math/prepare_data.py          # -> datasets/dapo-math-17k.parquet (17,917 problems)

sbatch math/train.sbatch                    # DTDD: tau = 0.362, 128-token windows, 120 updates
DTDD_ENABLE=0 sbatch math/train.sbatch      # OPD baseline, identical settings
SMOKE=1 sbatch --time=2:00:00 math/train.sbatch   # 2-update check

ARM=dtdd_math CKDIR=$DTDD_ROOT/checkpoint/dtdd_math STEPS="60 80 100 120" \
  sbatch math/eval/eval.sbatch               # average@16 on AIME24, AIME25, AMC23
```

On 4 GPUs the student trains on two and two dedicated vLLM engines generate the
teacher suffixes on the other two. On 44–48 GB cards the update pass uses
Ulysses sequence parallelism, which needs the small fix in
`math/verl_sp_fix.patch` (applied by `setup.sh`).

## ALFWorld

```bash
alfworld/make_venv.sh $DTDD_ROOT/alfworld_venv && export AGENTIC_VENV=$DTDD_ROOT/alfworld_venv   # alfworld/README.md
export ALFWORLD_DATA=$DTDD_ROOT/datasets/alfworld
$AGENTIC_VENV/bin/alfworld-download && ./setup.sh --alfworld   # points the task lists at the game files
huggingface-cli download Qwen/Qwen2.5-3B-Instruct                      --local-dir models/Qwen2.5-3B-Instruct
huggingface-cli download langfeng01/GiGPO-Qwen2.5-7B-Instruct-ALFWorld --local-dir models/GiGPO-Qwen2.5-7B-Instruct-ALFWorld

CONFIG=alfworld/configs/dtdd_lora.yaml sbatch alfworld/train.sbatch   # DTDD, tau = 0.7, 400 steps
CONFIG=alfworld/configs/opd_lora.yaml  sbatch alfworld/train.sbatch   # OPD baseline

ARM=dtdd_lora_alfworld STEPS="400" CONFIG=alfworld/configs/dtdd_lora.yaml \
  sbatch alfworld/eval.sbatch                # full seen + unseen test sets
```

## Coding

```bash
code/make_env.sh $DTDD_ROOT/code_env && export CODE_ENV=$DTDD_ROOT/code_env   # code/README.md
# data: training slice, 200-problem held-out split, LiveCodeBench v1/v6 (code/README.md)
# sandbox: a SandboxFusion server runs every program (code/README.md)
huggingface-cli download Qwen/Qwen3-1.7B                    --local-dir models/Qwen3-1.7B
huggingface-cli download youngzhong/SOD-GRPO_teacher-4B     --local-dir models/SOD-GRPO_teacher-4B
huggingface-cli download --repo-type dataset Gen-Verse/Open-AgentRL-SFT-3K \
  --local-dir datasets/code/Open-AgentRL-SFT-3K

sbatch code/sft.sbatch                        # the student: cold-start SFT -> models/Qwen3-1.7B-SOD-SFT
ARM=dtdd sbatch code/train.sbatch             # DTDD, tau = 0.50
ARM=opd  sbatch code/train.sbatch             # OPD baseline (same code path, trigger disabled)

ARM=qwen3_1p7b_code_dtdd STEPS="60" sbatch code/eval.sbatch   # held-out + LiveCodeBench, average@8
```

## Results

The main findings of the paper. Accuracy is average@k (%) on held-out sets;
agentic numbers are success rates (%).

### 1. DTDD removes the long-horizon OPD instability

DeepSeek-R1-Distill-Qwen-1.5B ← JustRL-DeepSeek-1.5B on DAPO-Math-17K. With
15K and 18K response budgets, OPD enters a late gradient storm together with a
spike in truncated rollouts. DTDD cuts the peak gradient norm **8.6×**
(12.02 → 1.40) and **7.0×** (12.81 → 1.84), with held-out accuracy within one
point of OPD at every budget.

<p align="center"><img src="assets/stability.png" width="92%" alt="Gradient norm and truncated-rollout ratio over training at 15K and 18K budgets: OPD spikes near update 200, DTDD stays low."></p>

### 2. DTDD warms up a base student online, without an SFT stage

Qwen3-1.7B-Base ← Qwen3-4B-Base-GRPO on DAPO-Math-17K. DTDD reaches the
21.3% held-out reference level near **update 70**; OPD needs about **180**.
Counting wall-clock time, DTDD reaches 21.34% in 22.0 hours, while offline
SFT followed by OPD reaches 21.55% in about 42.8 serial hours, using 95M
teacher-written tokens against DTDD's 43M. Both later converge near 22%.

<p align="center"><img src="assets/base_warmup.png" width="92%" alt="Held-out AIME24, AIME25 and AMC23 accuracy over training: DTDD with 64-, 128- and 256-token windows rises well before OPD."></p>

### 3. Both design choices matter: segment-level triggering and BC on the suffix

Same base-student setting, update 120 (OPD and DTDD: two-seed means). The math
example in this repository is the DTDD row.

| Trigger | Suffix loss | Teacher tokens / update | Accuracy |
|---|---|---|---|
| none (OPD) | – | 0 | 14.54 |
| **Segment (DTDD)** | **BC** | 0.52M | **20.83** |
| Token spike | KD | 0.40M | 13.51 |
| Token spike | BC | 0.43M | 16.79 |
| Segment | KD | 0.67M | 10.23 |
| AdaSwitch | KD | 0.86M | 19.95 |

### 4. It transfers to multi-turn agents

Best-observed checkpoints; takeover is used only in training, and evaluation
runs the student alone.

| Setting | OPD | DTDD |
|---|---|---|
| ALFWorld: Qwen2.5-3B-Instruct (LoRA) ← GiGPO-7B, seen / unseen | 83.57 / 69.40 | **84.29 / 78.36** |
| ALFWorld average | 76.49 | **81.32** (+4.83) |
| Coding: Qwen3-1.7B (SFT) ← 4B GRPO teacher, average of held-out and LiveCodeBench | 36.49 | **40.59** (+4.10) |

On ALFWorld, takeover starts at 90% and falls to near zero as the student
aligns, then briefly re-activates when the discrepancy rises again, so DTDD
also works as an online monitor. On coding, a sparse 2.8% teacher-token dose
improves all four evaluation splits.

### 5. When takeover helps: headroom

Across 23 arms in 13 settings, 19 follow one pattern: DTDD's gains concentrate
where plain OPD leaves headroom below its teacher,
h = 1 − OPD<sub>end</sub> / teacher. Three controls mark the limits. The
teacher must be a valid source at the takeover state (on WebShop it scores
below OPD, and takeover hurts); the student must be able to learn the recovery
(a 0.5B ALFWorld student cannot); and the dose must stay moderate (about 50%
takeover on coding costs 5.8 points).

<p align="center"><img src="assets/headroom.png" width="80%" alt="DTDD minus OPD against headroom left by OPD: gains appear at moderate-to-large headroom, losses where OPD is near or above the teacher."></p>

## Repository layout

```
setup.sh        fetch the upstream trainers at their pinned commits, apply the patches
math/           DTDD on thunlp/OPD (verl): trainer, takeover, teacher pool, launcher, eval
alfworld/       DTDD on kokolerk/TCOD (trinity): workflow, configs, harness patch, launchers
code/           DTDD on YoungZ365/SOD (verl agent loop): recipe, patches, data builders, sandbox
assets/         figures
```

## Citation

```bibtex
@article{he2026dtdd,
  title  = {DTDD: Divergence-Triggered Dynamic Distillation for Reliable On-Policy Supervision},
  author = {He, Qijia and Li, Zequn and Luo, Spencer and Chen, Tianle and Narayana, Pradyumna and
            Ma, Zixian and Krishna, Ranjay and Nie, Lan and Basu, Sugato},
  year   = {2026}
}
```

## Acknowledgements

DTDD is built on [thunlp/OPD](https://github.com/thunlp/OPD) and
[verl](https://github.com/volcengine/verl) (math),
[TCOD](https://github.com/kokolerk/TCOD) / Trinity-RFT and
[ALFWorld](https://github.com/alfworld/alfworld) (ALFWorld), and
[SOD](https://github.com/YoungZ365/SOD),
[SandboxFusion](https://github.com/bytedance/SandboxFusion) and
[LiveCodeBench](https://livecodebench.github.io/) (coding). We use the
released Qwen3, Qwen2.5, GiGPO (verl-agent) and SOD checkpoints, and the
DAPO-Math-17K and Open-AgentRL datasets.

## License

Apache-2.0 (see `LICENSE`). The upstream trainers fetched by `setup.sh` keep
their own licenses; `code/recipe/dtdd/*.py` are derived from verl
(Copyright Bytedance Ltd., Apache-2.0). See `NOTICE`.
