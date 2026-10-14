# What the TCOD overlay patch changes, and why

`tcod_overlay.patch` applies to `kokolerk/TCOD@465eef4` (the pinned base that
`setup.sh` fetches into `third_party/TCOD`). It touches six files, adds 127
lines and removes 26. **None of it is the DTDD method.** The method lives
entirely in the two workflow files this directory ships alongside
(`workflows/`); the patch is the harness hardening that was needed to run
multi-day LoRA and full-fine-tuning jobs on a shared cluster without the
run dying or, worse, silently training on a biased batch.

Each hunk below names the failure it fixes. The patch is written to apply
with `git apply`; `setup.sh` checks that it still does and fails loudly if
the base has moved.

| File | Lines | Fixes |
|---|---|---|
| `trinity/buffer/utils.py` | +44 −25 | a retry helper that could not retry |
| `trinity/common/config.py` | +8 −1 | Ray actors not inheriting cache / W&B settings |
| `trinity/common/models/vllm_model.py` | +48 | SIGBUS from loading a half-written LoRA adapter |
| `trinity/explorer/explorer.py` | +12 −1 | resume leaving the rollout engine on step-0 weights |
| `trinity/utils/lora_utils.py` | +12 | SIGBUS from the dummy adapter on a shared filesystem |
| `trinity/common/workflows/__init__.py` | +1 | registers the DTDD ALFWorld workflow |

## `trinity/buffer/utils.py` — `retry_session`

The upstream context manager put its `yield` inside a `for` retry loop. A
`@contextmanager` generator must yield exactly once; on any commit failure
the retry made it yield a second time, which `contextlib` turns into
`RuntimeError: generator didn't stop`. That masked the real error — a
transient sqlite `database is locked` under Ray's concurrent writers — and
crashed the explorer.

A second defect sat underneath: a failed commit leaves the SQLAlchemy
session in a pending-rollback state, and upstream only rolled back on the
*final* attempt, so attempt 2 raised `PendingRollbackError` instead of
retrying. The fix yields once, retries only the commit, rolls back before
every retry, and backs off exponentially — lock contention needs more than
a flat wait.

## `trinity/common/config.py` — `Config.get_envs`

Ray actors do not inherit the driver's environment. Upstream forwarded only
its own logging variables, so the HF cache and W&B settings set in the
launcher never reached the workers, which then wrote to the default `~/.cache`
— a quota-limited home directory on most clusters. The hunk forwards
`HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE` and the
`WANDB_*` variables when they are set. It changes nothing when they are not.

## `trinity/common/models/vllm_model.py` — `_wait_adapter_complete`

LoRA weight sync in this harness is filesystem-based: the trainer writes
`global_step_N/actor/lora_adapter` and the vLLM engine `mmap`-loads it. An
`mmap` of a file that is still being written raises `SIGBUS`, which kills the
engine-core child process outright and cannot be caught, leaving the job a
zombie. In steady state the sync handshake orders write-before-read; on
**resume** the first sync races the trainer's re-save of the step-N adapter,
and on a shared filesystem a ~240 MB adapter takes one to two minutes to
land. This was reproducible on every resume of a LoRA run.

A safetensors file self-describes its full length (an 8-byte header size
plus a JSON header carrying data offsets), so completeness can be checked
exactly rather than guessed from file size. The hunk waits until the header
parses, the declared payload fits in the file, and the size is stable across
one poll, before handing the path to vLLM. `sync_model` calls it before every
adapter swap.

## `trinity/explorer/explorer.py` — first-sync guard

Upstream skipped the weight push when `self.model_version == -1`, i.e. on the
first sync of the process. On a fresh run that is correct: the new version is
0 and the engine already holds those weights. On a **resume** the engine still
holds step-0 weights while the trainer is at step N, and skipping the push
leaves the rollout engine at `model_version=0` forever. Every experience is
then tagged version 0, the trainer's staleness filter discards the whole
batch, and the run dies with `Read incomplete batch (0/N)` — after sampling
from an untrained adapter path that does not exist. The hunk pushes whenever
there is a real (`> 0`) version to push.

## `trinity/utils/lora_utils.py` — `TRINITY_DUMMY_LORA_DIR`

`create_dummy_lora` writes a ~240 MB placeholder adapter at every startup
that the vLLM engine `mmap`-loads minutes later. On a shared filesystem that
page-in intermittently died with `SIGBUS` (always in `add_lora ->
from_local_checkpoint`; the file verified fine afterwards; neither excluding
nodes nor staggering starts helped). The dummy is throwaway state only this
node's engine ever reads, so the hunk lets `TRINITY_DUMMY_LORA_DIR` redirect
it to node-local storage such as the job's `/tmp`. Default behaviour is
unchanged when the variable is unset; the shipped launchers set it.

## `trinity/common/workflows/__init__.py` — registry

One line registering `DTDD_alfworld_workflow` by the module path `setup.sh`
installs it at. This is the name the YAML configs reference in
`default_workflow_type`.
