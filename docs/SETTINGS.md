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
