# Experiment settings

All settings share: GRPO-style token-level distillation reward on the
student's own tokens (top-16 teacher/student probability comparison,
`only_stu` strategy), teacher-written takeover suffixes learned by imitation,
`data.shuffle=False` for paired arms, held-out eval = decoupled vLLM avg@16 at
31744 tokens on AIME24 / AIME25 / AMC23.

| setting | student ← teacher | data | len | τ / mode | steps | launch |
|---|---|---|---|---|---|---|
| 7K (no-harm control) | R1-Distill-1.5B ← JustRL-1.5B | DAPO-Math-17k | 7168 | 0.5 rewind | 279 | `settings/submit_18k.sh` with `MAX_RESP_LENGTH=7168` |
| 12K / 15K / 18K (length sweep) | same pair | DAPO-Math-17k | 12288 / 15360 / 18432 | 0.5 rewind | 279 | `settings/submit_18k.sh` (edit `MAX_RESP_LENGTH`; 15K+ needs A100-80G) |
| cold-start (paper §5.1) | Qwen3-1.7B-**Base** ← Qwen3-4B-GRPO | DAPO-Math-17k | 7168 | **0.2** rewind | 279 | `settings/submit_dtddA_taufix_rs.sh` |
| — const-rate ablation | same | same | 7168 | `DTDD_RATE=0.25` | ~130 | `settings/submit_dtddA_constrate_resume.sh` |
| — restart ablation (full demo) | same | same | 7168 | `DTDD_RATE=0.45`, mode `restart` | ~100 | `settings/submit_dtddA_restart.sh` |
| hard-gap | R1-Distill-1.5B ← JustRL, subset {student pass@4=0 ∩ teacher ≥ .5} | `scripts/data/tag_hard_pair.py` output | 7168 | 0.5 rewind | 150 | `settings/submit_hardgap.sh` |
| SFT-mismatch | Qwen3-1.7B-SFT ← Qwen3-4B-GRPO | DAPO-Math-17k | 7168 | 0.5 rewind | 279 | (env swap on the cold-start script) |
| 0.6B dead-zone | Qwen3-0.6B-Base ← Qwen3-4B-GRPO | DAPO-Math-17k | 8192 | 0.5 rewind | ~30 | (env swap) |
| StableOPD replication | R1-Distill-1.5B ← R1-Distill-**7B** | DAPO-Math-17k | 7168 | **0.5** rewind (calibrated: would-trigger ~9%→3%) | 279 | `settings/submit_stableopd.sh` |
| collapse2 (negative) | Qwen2.5-Math-1.5B ← JustRL | DAPO-Math-17k | **3072** (cap artifact — do not reuse) | 0.5 | — | `settings/submit_collapse2.sh` |
| OPD dense reference | Qwen3-1.7B-Base ← Qwen3-4B-GRPO | DAPO-Math-17k | 7168 | (OPD) | 279 | `settings/submit_opd_from_base_dense.sh` |

Calibration guidance for a new pair: run a ~10-step dry-run and pick τ from the
`would_trigger@τ` sweep — a healthy tripwire fires on ~5–10% of windows at
step 1 and decays; 25%+ is a constant-dose regime, ~1% is deaf. Non-same-family
teachers with a larger embedding than the student need the sampling vocab cap
(`teacher_pool.py: sample_vocab_cap`, e.g. 151665 for Qwen2.5-family teachers
with Qwen3 students).

## Agentic settings (ALFWorld, WebShop)

All agentic settings share: multi-turn on-policy distillation on the student's
own action tokens (`multi_turn_opd` advantage, `kl_coef 1.0`), one takeover
decision **per environment turn** on the divergence
`mean_t |logp_student - logp_teacher|` over that turn's action tokens, teacher
turns learned by imitation, evaluation always student-only, held-out eval =
full seen/unseen (ALFWorld, 140/134 episodes) or the 2,000 held-out goals
(WebShop) at temperature 0.4. The base harness is TCOD's trinity runner
(`third_party/TCOD`); see `recipe_dtdd_agentic/overlay/PATCH_NOTES.md` for the
harness fixes every arm needed.

| setting | student ← teacher | data | τ / min prefix | steps | config (`recipe_dtdd_agentic/configs/`) |
|---|---|---|---|---|---|
| ALFWorld LoRA (paper Table 1) | Qwen2.5-3B-Instruct (LoRA r32) ← GiGPO-7B-ALFWorld | ALFWorld `train_expert` | **0.7** / 1 | 400 (~1.8 epochs) | `alfworld/dtdd_lora.yaml` vs `alfworld/opd_lora.yaml` |
| ALFWorld full-FT | Qwen2.5-3B-Instruct (full) ← same | same | 0.7 / 1 | 400 | `alfworld/dtdd_fullft.yaml` vs `alfworld/opd_fullft.yaml` |
| ALFWorld 14× gap (boundary case) | Qwen2.5-**0.5B**-Instruct (LoRA) ← same 7B | same | 0.7 / 1 | 200 | `alfworld/dtdd_gap14x.yaml` vs `alfworld/opd_gap14x.yaml` |
| WebShop 120k (teacher-validity case) | Qwen2.5-3B-Instruct (LoRA) ← GiGPO-7B-WebShop | 120k-product catalog, 10,251 / 2,000 goals | **0.25** / 1 | 200 | `webshop/dtdd_lora.yaml` vs `webshop/opd_lora.yaml` |

Each DTDD config differs from its OPD partner only in `default_workflow_type`
(`DTDD_*_workflow` vs `OPD_*_workflow`) and the `tau` entry under
`workflow_args`; everything else — student, teacher, data, LoRA rank, batch
sizes, eval schedule — is identical, so the pairs are matched arms.

**Why τ differs between environments.** τ is a threshold on a mean absolute
log-prob gap, and that gap's scale depends on the action space. ALFWorld
actions are short imperative strings from a small admissible set; WebShop
actions embed free-text search queries and item ids, so per-token gaps run
smaller and 0.7 would almost never fire. 0.25 was chosen from a dry-run
`would_trigger@τ` sweep on WebShop the same way the math lane's τ is: a
healthy tripwire fires on a minority of turns at step 1 and decays. The
resulting takeover-rate curves are in the paper (Figure 5 for ALFWorld).

**Steps.** The LoRA and full-FT ALFWorld runs were extended from 200 to 400
updates (about 1.8 epochs of `train_expert`; the reader wraps) to check the
second-epoch behaviour reported in the paper; the configs ship at 400. The
14× gap and WebShop arms stop at 200.

**WebShop teacher.** In the rebuilt 120k environment the GiGPO-7B WebShop
teacher scores 18.7% success on the held-out goals, *below* the OPD student's
19.7% at update 200 — the "invalid teacher" boundary case. Score the teacher
yourself with `scripts/eval/run_bench_agentic.sbatch` (`STEPS=0`,
`STUDENT_MODEL=<teacher>`) before drawing any conclusion from a WebShop arm.

## Code settings (LiveCodeBench)

The code lane trains a tool-using coding agent: every response is executed
against test cases in a SandboxFusion instance, and that pass rate is the
reward. It shares the trigger, the takeover and the loss with the other
lanes, but rides on a different base (`YoungZ365/SOD`, itself a fork of
verl) and a different environment; see
`recipe_dtdd_code/overlay/PATCH_NOTES.md`.

All code settings share: student Qwen3-1.7B-SFT ← teacher
`SOD-GRPO_teacher-4B` (the SOD release), the code-only slice of
Open-AgentRL-30K (4,205 problems, grading tests subsampled to ≤12), up to 8
tool-calling turns, 12,288 response tokens, batch 64 × 8 samples,
`token-mean` aggregation, lr 1e-6, about 66 optimizer steps per epoch
(4,205 problems / batch 64) and 2 epochs, so the arms run to step 131;
checkpoints and validation every 10 steps. Rollouts are student-only at
evaluation time. 4 GPUs: student on two (TP=2, utilisation 0.45), the frozen
teacher pool on the other two (dp=2, TP=1, utilisation 0.8) via
`DTDD_TEACHER_GPU_OFFSET=2`.

| setting | τ / trigger | ε random | steps | how to launch |
|---|---|---|---|---|
| DTDD (paper Table, code block) | **0.50** rewind | 0.2 | 131 (2 epochs) | `ARM=dtdd sbatch scripts/train/run_dtdd_code.sbatch` |
| DTDD high dose | **0.26** rewind | 0.2 | 131 | `ARM=dtdd DTDD_TAU=0.26 EXPERIMENT_NAME=... sbatch ...` |
| OPD baseline (matched) | 999 — never fires | 0 | 131 | `ARM=opd sbatch scripts/train/run_dtdd_code.sbatch` |

**The baseline is the same binary.** It keeps `dtdd.enable=true` and sets the
threshold out of reach, so the two arms share the rollout path, the teacher
engine, the reference logprobs and the loss code exactly; the only
difference is whether the trigger ever fires. `STEPWISE_ENABLE=False` on
both arms is the other half: the step-wise `w_k` weighting is SOD's
contribution, not vanilla OPD, so turning it off makes "OPD" mean the same
thing here as in the math and agentic lanes.

**Why τ differs from the other lanes.** τ is a threshold on a mean absolute
log-prob gap per turn, and its scale depends on what a turn contains. A code
turn is a long block of program text plus a tool call; the math lane's
reasoning steps and ALFWorld's short imperative actions sit at different
scales, so the numbers are not comparable across lanes. 0.50 is roughly the
75th percentile of the divergence measured on a vanilla-OPD run
(`scripts/calibrate_tau.py` in the SOD base reports that distribution);
0.26 is the deliberate over-dose, and it is the arm that comes out negative.

**Evaluation.** `scripts/eval/run_eval_code.sbatch` scores saved checkpoints
on four sets in one pass: the 200 held-out training-distribution problems
with subsampled tests, the same 200 with their full suites, LiveCodeBench v1
(219 problems, 2023-05..2024-03) and v6 (112 problems, 2025-01..2025-04).
Reading accuracy off this fixed harness rather than off the training curves
is deliberate: the runs log a partial-credit `score`, not `acc`, and the two
correlate at r = 0.30.

**The LiveCodeBench files are rebuilt, not the shipped ones.** The parquet
in `Open-AgentRL-Eval` grades each problem on the single test case it also
prints into the prompt, so echoing the shown output scores 1.0. Build the
split versions with `recipe_dtdd_code/data/build_lcb_fixed.py` before
drawing any conclusion from a LiveCodeBench number.

**Sandbox.** One SandboxFusion instance per training job
(`scripts/sandbox/`). A single shared instance serving several concurrent
trainings is both a single point of failure and the overload pattern that
produces sick-but-alive 503s; the launchers start a dedicated one and cancel
it on exit. If you cancel a training job by hand, cancel its
`sandbox_<jobid>` companion too — it does not exit on its own.

**Three settings that are not free parameters.**

* **The teacher needs its own GPUs** (`DTDD_TEACHER_GPU_OFFSET=2`). Sharing
  devices with the student has no working operating point: a low teacher
  utilisation starves its KV cache, until scoring is nearly all of a step; a
  high one makes the student's engine fail to re-map its memory on wake.
* **`ppo_max_token_len_per_gpu=8192`, not prompt + response.** The vocabulary
  logits gradient in the update's backward pass is the peak allocation of the
  whole step, and the full-length value overflows a 48 GB card. This is legal
  because the value times the sequence-parallel degree still exceeds the
  longest sequence.
* **Never set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`.** It conflicts
  with the vLLM weight sync, and the first synchronisation hangs with no
  error.

**Resuming.** Checkpoints resume automatically
(`trainer.resume_mode=auto`): after a timeout or preemption, resubmit the
same command. Chain with `--dependency=afterany` rather than retrying inside
one allocation — an in-job retry reuses the allocation, where a crashed
engine still holds its memory and the student cannot allocate a KV cache.
