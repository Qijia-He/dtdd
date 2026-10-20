# ALFWorld example

DTDD on [TCOD](https://github.com/kokolerk/TCOD) (Trinity-RFT). The method is
the two workflow files in `workflows/`; `tcod_overlay.patch` is harness
hardening for multi-day LoRA runs (see `PATCH_NOTES.md`), not part of the
method.

| File | What it is |
|---|---|
| `workflows/dtdd_base_workflow.py` | turn-level divergence, trigger, teacher takeover, per-turn records |
| `workflows/dtdd_alfworld_workflow.py` | the ALFWorld environment loop |
| `configs/dtdd_lora.yaml`, `configs/opd_lora.yaml` | the two paper arms (LoRA, tau = 0.7, 400 steps) |
| `train.sbatch`, `eval.sbatch` | launchers (training; full held-out benchmark of saved checkpoints) |
| `make_venv.sh`, `requirements-venv.txt`, `requirements-lock.txt` | the environment: the exact package layer it installs over the image, and the full freeze |

## Environment

The same `verlai/verl:vllm011` image as the math example, plus a venv layered
over it: TCOD needs a newer torch, vLLM and Ray than the image carries, and
ALFWorld/TextWorld, which it lacks. The image itself stays unmodified.

| | |
|---|---|
| Venv | `make_venv.sh <dir>` (runs inside the container, so start it from a compute node); installs the exact layer in `requirements-venv.txt`; full freeze in `requirements-lock.txt` |
| Overrides | torch 2.8.0 → **2.9.1** · vLLM 0.11.0 → **0.14.1** · Ray 2.49.2 → **2.56.1** · flash-attn 2.8.1 rebuilt for torch 2.9 · numpy 1.26.4 · scipy 1.13.1 |
| Added | alfworld 0.4.2 · textworld 1.7.0 · peft 0.17.1 |
| Editable | `trinity_rft` → `third_party/TCOD` (the patched checkout is what gets imported) |

The pins that matter, in order of risk: `vllm==0.14.1` (TCOD's supported
ceiling; 0.12.x regresses badly), `torch==2.9.1`, `flash_attn==2.8.1` built
against it, `ray==2.56.1`.

## Data

The task lists ship with the TCOD base; the game files do not. Download them
once and point the task lists at them:

```bash
export ALFWORLD_DATA=$DTDD_ROOT/datasets/alfworld
$AGENTIC_VENV/bin/alfworld-download   # -> $ALFWORLD_DATA/json_2.1.1, logic/, detectors/
./setup.sh --alfworld                 # rewrites the task lists' game-file prefix
```

`setup.sh` regenerates the lists from the pristine upstream copies each time,
so changing `ALFWORLD_DATA` and re-running is safe.

## Models

```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct                      --local-dir models/Qwen2.5-3B-Instruct
huggingface-cli download langfeng01/GiGPO-Qwen2.5-7B-Instruct-ALFWorld --local-dir models/GiGPO-Qwen2.5-7B-Instruct-ALFWorld
```

## Run

```bash
CONFIG=alfworld/configs/dtdd_lora.yaml sbatch alfworld/train.sbatch   # DTDD
CONFIG=alfworld/configs/opd_lora.yaml  sbatch alfworld/train.sbatch   # OPD baseline

ARM=dtdd_lora_alfworld STEPS="400" CONFIG=alfworld/configs/dtdd_lora.yaml sbatch alfworld/eval.sbatch
```

Training resumes automatically after a timeout or preemption (resubmit the same
command). The evaluation merges each LoRA checkpoint onto the base student,
runs the full seen and unseen test sets, and writes
`agent_ckpts/<project>/<arm>_bench_<step>/log/benchmark.log`.
