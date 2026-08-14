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
