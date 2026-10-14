# Coding example

DTDD on [SOD](https://github.com/YoungZ365/SOD)'s verl agent loop: multi-turn
code generation with a Python sandbox tool, graded by running the program.
The method is `recipe/dtdd/`; `overlay/sod_overlay.patch` adds LiveCodeBench
support and robustness fixes to the base (see `overlay/PATCH_NOTES.md`).

| File | What it is |
|---|---|
| `recipe/dtdd/dtdd_agent_loop.py` | turn-level divergence, trigger, teacher takeover |
| `recipe/dtdd/dtdd_dp_actor.py` | OPD on the student prefix, cross-entropy on the teacher suffix |
| `recipe/dtdd/teacher_vllm_server.py`, `dtdd_ray_trainer.py`, `dtdd_fsdp_workers.py` | dedicated teacher engines and trainer wiring |
| `recipe/dtdd/run_dtdd_code.sh`, `config/` | the hydra overrides behind the paper's runs |
| `sft.sbatch` | builds the student (SOD's cold-start SFT) |
| `train.sbatch`, `eval.sbatch` | launchers |
| `data/` | builders for the training slice, the HO-200 evaluation split and LiveCodeBench |
| `sandbox/`, `overlay/sandbox_runner_stdin.patch` | the SandboxFusion server every program runs in |

## Environment

Its own conda environment, not the container: the SOD base pins
`vllm<=0.9.1`, older than the image's stack.

```bash
code/make_env.sh $DTDD_ROOT/code_env && export CODE_ENV=$DTDD_ROOT/code_env
```

| | |
|---|---|
| Interpreter | conda, Python 3.11 |
| Key pins | torch 2.6.0+cu124 · vLLM 0.8.5.post1 · ray 2.47.1 · flash-attn 2.7.4.post1 · numpy 1.26.4 |
| Full freeze | `requirements-lock.txt` |
| Editable | `verl` → `third_party/SOD` (the patched checkout is what gets imported) |

Two wheels are not on PyPI for this torch and CUDA combination
(`flash_attn 2.7.4.post1` and the optional `flashinfer-python 0.2.2.post1`);
the script fetches them from the projects' release pages. Keep the tags exact:
a wheel built against a different torch imports fine and then segfaults.

## Models

```bash
huggingface-cli download Qwen/Qwen3-1.7B                --local-dir models/Qwen3-1.7B
huggingface-cli download youngzhong/SOD-GRPO_teacher-4B --local-dir models/SOD-GRPO_teacher-4B
huggingface-cli download --repo-type dataset Gen-Verse/Open-AgentRL-SFT-3K --local-dir datasets/code/Open-AgentRL-SFT-3K
sbatch code/sft.sbatch     # student: 5-epoch SFT (115 updates) -> models/Qwen3-1.7B-SOD-SFT
```

## Data

```bash
huggingface-cli download --repo-type dataset Gen-Verse/Open-AgentRL-30K --local-dir datasets/Open-AgentRL-30K

# 1. the training slice and the 200-problem HO-200 evaluation split
$CODE_ENV/bin/python code/data/build_code_slices.py \
    --source datasets/Open-AgentRL-30K/Open-AgentRL-30K.parquet \
    --indices code/data/heldout200_indices.json \
    --output-dir datasets/code

# 2. LiveCodeBench v1 and v6, rebuilt so the prompt's example is not the grading test
for spec in "test.jsonl v1" "test6.jsonl v6"; do
  set -- $spec
  $CODE_ENV/bin/python code/data/build_lcb_fixed.py \
      --release "$1" --data-source "LiveCodeBench_$2_fixed" \
      --output "datasets/code/lcb_$2_fixed.parquet"
done
```

The training slice keeps every `train-code-*` row of Open-AgentRL-30K (4,205
problems) and caps each one's grading tests at 12, since every training
response is executed. HO-200 ships as the exact 200 row indices. These 200
problems are part of `code_slice_train.parquet`, which the paper's runs (and the
default launcher) train on, so HO-200 measures in-distribution performance on
seen problems; LiveCodeBench is the held-out benchmark. To keep HO-200 unseen,
train on `code_slice_train_minus200.parquet` (written by the same command):
`OPEN_AGENT_RL=datasets/code/code_slice_train_minus200.parquet`.
The LiveCodeBench files are rebuilt because the parquet in
`Open-AgentRL-Eval` uses one test case both as the worked example in the
prompt and as the grading truth, so a program that echoes the shown output
scores 1.0; the rebuilt files keep the two apart. The raw releases come from
`livecodebench/code_generation_lite`; some are over a gigabyte, so build one
release at a time on a machine with real memory.

## Sandbox

Rewards come from running the model's programs in a
[SandboxFusion](https://github.com/bytedance/SandboxFusion) server:

```bash
apptainer pull sandbox-fusion.sif docker://volcengine/sandbox-fusion:server-20250609
export SANDBOX_SIF=$PWD/sandbox-fusion.sif
apptainer exec "$SANDBOX_SIF" cat /root/sandbox/sandbox/runners/base.py > sandbox_base_patched.py
patch sandbox_base_patched.py code/overlay/sandbox_runner_stdin.patch
```

`sandbox/acquire_sandbox.sh`, sourced by both launchers, starts one instance
per job and cancels it on exit; set `SANDBOX_FUSION_URL` to reuse an existing
endpoint instead. The patch matters: the published image crashes when it kills
a timed-out process and afterwards answers health checks while every execution
fails.

## Run

```bash
ARM=dtdd sbatch code/train.sbatch     # DTDD, tau = 0.50 (random-trigger rate 0.2)
ARM=opd  sbatch code/train.sbatch     # OPD baseline: same code path, trigger disabled (tau = 999)
ARM=qwen3_1p7b_code_dtdd STEPS="60" sbatch code/eval.sbatch
```

The evaluation scores HO-200 (subsampled and full test suites) and
LiveCodeBench v1/v6 at average@8 and appends rows to
`results/code_eval.jsonl`.
