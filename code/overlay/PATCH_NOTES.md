# Code-lane overlay: what each hunk changes and why

Two patches, applied by `setup.sh --code`:

* **`sod_overlay.patch`** — 18 hunks over 8 files of `YoungZ365/SOD@110c4b8`.
  None of them is the method: DTDD itself lives entirely in
  `code/recipe/dtdd/`, which is copied in, not patched. These are
  an evaluation-correctness fix, a metrics fix, and four robustness fixes that
  every long run needed.
* **`sandbox_runner_stdin.patch`** — 1 hunk against the SandboxFusion server
  image, not against SOD. See the last section.

## `recipe/demystify/reward.py` — 2 hunks (evaluation correctness)

`map_lcb` printed a problem's test case into the prompt as the worked example
**and** handed the same case to `compute_score` as the grading truth. The
LiveCodeBench parquet shipped with `Open-AgentRL-Eval` carries exactly one
case per problem (median = mean = max = 1 across all 175), so a program that
ignored the question and echoed the shown output scored 1.0.

* Read the prompt example from a new `public_tests` column when it is
  present, keeping `reward_model` for grading only. When the column is
  absent the old behaviour is used unchanged, so the original files still
  load and their numbers stay reproducible.
* Route by `data_source.startswith("LiveCodeBench")` instead of an exact
  match on `LiveCodeBench_v6`, so the rebuilt per-release files
  (`LiveCodeBench_v1_fixed`, `..._v6_fixed`) reach the same map function.

`code/data/build_lcb_fixed.py` writes the split parquets.

## `verl/trainer/ppo/ray_trainer.py` — 3 hunks (metrics only)

* `_write_stepwise_log` returns a summary dict (`opd/avg_steps`,
  `opd/avg_d1`, `opd/stepwise_weight_{mean,min,max}`) instead of only writing
  a log file, and `_apply_token_kl_regularizer` passes it through
  `meta_info["token_kl_reg"]["stepwise_summary"]` into the step metrics. The
  OPD step-weight distribution was otherwise visible only by parsing a text
  file after the fact. The DTDD trainer reads the same channel.
* `training/epoch` becomes fractional (`epoch + (step + 1) / len(loader)`).
  The integer version cannot say where inside an epoch a checkpoint sits,
  which is exactly what the paper's step-to-epoch statements need.

No hunk touches the loss, the advantage or the optimizer.

## `verl/trainer/constants_ppo.py` — 1 hunk (robustness)

Forces `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR` and `VLLM_CACHE_ROOT`
into the Ray runtime environment, defaulting to node-local storage. Plain
environment inheritance into Ray `WorkerDict` actors is not reliable for
these, and on a full or quota-bound home directory every rank hit
`OSError: [Errno 122] Disk quota exceeded` while compiling a kernel — with
no crash and no exit, so the job sat in RUNNING for hours doing nothing.

## `verl/utils/checkpoint/fsdp_checkpoint_manager.py` — 1 hunk (robustness)

Resume with fresh optimizer state, and a warning, when a checkpoint's
optimizer shard is missing, instead of raising. Long runs on quota-bound
filesystems prune optimizer shards from intermediate checkpoints; refusing
to resume turns a recoverable run into a lost one, while fresh Adam state
costs a brief warmup transient.

## `verl/utils/model.py`, `verl/model_merger/base_model_merger.py` — 5 hunks

`transformers >= 4.54.0` renamed `AutoModelForVision2Seq` to
`AutoModelForImageTextToText`, and the unconditional top-level import fails
at import time on newer versions. Both files now import the new name behind a
version check and alias it. Pure compatibility; no behaviour change on either
version.

## `verl/utils/reward_score/livecodebench/code_math.py` — 3 hunks

Replaces the two hardcoded `"<your_sandbox_fusion_url>"` placeholders with a
module-level `SANDBOX_FUSION_URL` read from the environment (defaulting to
`http://localhost:8080/run_code`). Upstream expects you to edit the file;
reading the environment is what lets the launchers point each job at its own
sandbox instance. The tool config gets the same treatment through
`${oc.env:SANDBOX_FUSION_URL,...}`.

## `recipe/demystify/sandbox_fusion_tool_config.yaml` — 1 hunk

Same environment-driven URL, plus two limits: `num_workers` 32 → 16 (the
upstream default overwhelms a locally deployed sandbox under real
concurrency, which the SOD authors confirmed; match it to the sandbox
instance's CPU count) and `default_timeout` 60 → 10 s (training-time tool
calls are short scripts, and the 60 s ceiling let a few runaway executions
hold every worker).

## `sandbox_runner_stdin.patch` — the SandboxFusion image

Not a SOD patch. The `volcengine/sandbox-fusion:server-20250609` image
crashes when it kills a timed-out process: uvloop keeps writing to the dead
child's stdin, which raises, and the exception storm wedges the event loop —
the server then answers health checks while every execution fails. Upstream
fixed this in June 2025 but never published a newer image, so
`code/sandbox/start_server.sbatch` bind-mounts a patched copy of the
container's own `sandbox/runners/base.py` over the original. The patch is the
stdin write/close guard and nothing else.
