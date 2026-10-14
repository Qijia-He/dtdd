"""DTDD trainer for the Thinking-Space/Rethinking-OPD verl v0.7 fork.

Zero-copy insertion: subclass RayPPOTrainer, and in init_workers() wrap two
driver-side worker-group proxy methods (see PORT_PLAN.md for the full design):

  rm_wg.compute_rm_score      -> pass 1 (score) -> d_k windows -> first-passage
                                 trigger -> teacher continuation -> in-place
                                 splice -> refresh student logps -> pass 2;
                                 returns pass-2 teacher tensors (fit unions them).
  actor_rollout_wg.compute_distillation_reward
                              -> original, then exact-BC suffix override:
                                 slot-0 top-k id := teacher token, adv row :=
                                 [bc_coef, 0...]; sampled (2D) mode handled in
                                 the rm wrapper directly.

Config via DTDD_* env vars (no hydra schema changes). DTDD_ENABLE=0 makes this
trainer bit-identical to the stock one.
"""

import os

import numpy as np
import ray
import torch

from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from dtdd_takeover import compute_dk_from_diff, pick_triggers, splice_batch
from teacher_pool import TeacherGenPool


def _env_cfg():
    return {
        "enable": os.environ.get("DTDD_ENABLE", "0") == "1",
        "dry_run": os.environ.get("DTDD_DRY_RUN", "0") == "1",
        "tau": float(os.environ.get("DTDD_TAU", "0.7")),
        "chunk": int(os.environ.get("DTDD_CHUNK", "256")),
        "min_prefix_turns": int(os.environ.get("DTDD_MIN_PREFIX", "3")),
        "epsilon_random": float(os.environ.get("DTDD_EPS_RANDOM", "0.05")),
        "random_turn_max": int(os.environ.get("DTDD_RANDOM_MAX", "16")),
        "takeover_mode": os.environ.get("DTDD_MODE", "rewind"),
        "min_takeover_budget": int(os.environ.get("DTDD_MIN_BUDGET", "256")),
        "bc_coef": float(os.environ.get("DTDD_BC_COEF", "1.0")),
        "teacher_engines": int(os.environ.get("DTDD_TEACHER_ENGINES", "2")),
        "teacher_gpu_util": float(os.environ.get("DTDD_TEACHER_GPU_UTIL", "0.15")),
        "teacher_gpu_offset": int(os.environ.get("DTDD_TEACHER_GPU_OFFSET", "0")),
        # constant-rate mode: if >0, per-batch adaptive tau targets this takeover fraction
        "rate": float(os.environ.get("DTDD_RATE", "0")),
        "teacher_temperature": float(os.environ.get("DTDD_TEACHER_TEMP", "1.0")),
        "teacher_top_p": float(os.environ.get("DTDD_TEACHER_TOP_P", "1.0")),
    }


class DTDDRayPPOTrainer(RayPPOTrainer):
    def init_workers(self):
        super().init_workers()
        self.dtdd_cfg = _env_cfg()
        cfg = self.dtdd_cfg
        if not cfg["enable"]:
            print("DTDD disabled (DTDD_ENABLE!=1): stock trainer behavior")
            return

        rollout_cfg = self.config.actor_rollout_ref.rollout
        max_model_len = int(rollout_cfg.prompt_length) + int(rollout_cfg.response_length) + 8
        teacher_path = self.config.reward_model.model.path
        n_gpus = int(self.config.trainer.n_gpus_per_node)
        self._teacher_pool = []
        if cfg["dry_run"]:
            # Dry-run never takes over -> no teacher engines needed; saves ~10GB/GPU
            # so calibration fits on 44GB cards.
            print("DTDD dry-run: skipping teacher pool creation")
            self._wrap_workers(cfg)
            return
        # Student embedding size decides whether teacher sampling must be capped
        # (teacher embedding padded larger than the student's -> spliced ids could
        # index out of range; see the assert below).
        from transformers import AutoConfig
        _sv = AutoConfig.from_pretrained(
            self.config.actor_rollout_ref.model.path, trust_remote_code=True
        ).vocab_size
        _tv_cfg = AutoConfig.from_pretrained(teacher_path, trust_remote_code=True).vocab_size
        sample_cap = _sv if _tv_cfg > _sv else 0
        if sample_cap:
            print(
                f"DTDD: teacher embedding {_tv_cfg} > student {_sv}; "
                f"capping teacher sampling to ids < {_sv}"
            )
        for i in range(cfg["teacher_engines"]):
            actor = TeacherGenPool.options(
                runtime_env={
                    "env_vars": {
                        "DTDD_TEACHER_CVD": str(cfg["teacher_gpu_offset"] + (i % max(1, n_gpus))),
                        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    }
                }
            ).remote(
                teacher_path,
                {
                    "gpu_memory_utilization": cfg["teacher_gpu_util"],
                    "temperature": cfg["teacher_temperature"],
                    "top_p": cfg["teacher_top_p"],
                    "tensor_parallel_size": 1,
                    "enforce_eager": True,
                    "sample_vocab_cap": sample_cap,
                },
                max_model_len,
            )
            self._teacher_pool.append(actor)
        ray.get([p.ready.remote() for p in self._teacher_pool])
        # Teacher tokens get spliced into the student sequence and looked up in the
        # student embedding, so the teacher's sampling vocabulary must fit inside the
        # student's EMBEDDING size (padded config vocab_size), else embedding indexing
        # asserts device-side mid-training. Compare embedding sizes (config vocab_size),
        # NOT tokenizer length: tokenizers report the count of real tokens while the
        # embedding is padded up (e.g. 151665 tokens -> 151936-row embedding), so those
        # two legitimately differ for a valid same-family pair.
        _tv = ray.get(self._teacher_pool[0].vocab_size.remote())
        assert _tv <= _sv or sample_cap == _sv, (
            f"DTDD teacher can emit token ids up to its embedding size {_tv}, which "
            f"exceeds the student embedding size {_sv}; spliced teacher tokens would "
            f"index out of range. This teacher/student pair is invalid for takeover "
            f"(sampling cap not active)."
        )
        print(f"DTDD teacher pool ready: {cfg['teacher_engines']} engines, util {cfg['teacher_gpu_util']}")

        self._wrap_workers(cfg)

    def _wrap_workers(self, cfg):
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        rm_orig = self.rm_wg.compute_rm_score
        dist_orig = getattr(self.actor_rollout_wg, "compute_distillation_reward", None)
        actor_wg = self.actor_rollout_wg
        trainer = self

        def rm_wrapped(batch):
            out1 = rm_orig(batch)
            tb = batch.batch
            if "old_log_probs" not in tb.keys():  # validation path
                return out1
            R = tb["responses"].shape[1]
            P = tb["input_ids"].shape[1] - R
            chunk = cfg["chunk"]
            # per-position divergence
            if "teacher_on_student_log_probs" in out1.batch.keys():  # topk mode
                S = tb["student_top_k_log_probs"].float()
                T = out1.batch["teacher_on_student_log_probs"].float()
                w = torch.softmax(S, dim=-1)
                per_pos = ((S - T).abs() * w).sum(-1)
            else:  # sampled mode: rm_scores = T - S
                per_pos = out1.batch["rm_scores"].abs().float()
            rmask = tb["response_mask"].float()
            dk, cnt = compute_dk_from_diff(per_pos, rmask, chunk)
            step_seed = int(getattr(trainer, "global_steps", 0))
            cfg_eff = cfg
            if cfg["rate"] > 0:
                # adaptive tau: per-batch quantile of per-sequence max eligible d_k,
                # so the takeover fraction stays ~rate even as the student aligns.
                el = dk.clone()
                el[cnt <= 0] = float("-inf")
                el[:, : cfg["min_prefix_turns"] - 1] = float("-inf")
                seqmax = el.max(dim=1).values
                valid_rows = torch.isfinite(seqmax)
                if valid_rows.any():
                    tau_eff = float(torch.quantile(seqmax[valid_rows].float(), 1.0 - cfg["rate"]))
                    cfg_eff = dict(cfg)
                    cfg_eff["tau"] = tau_eff
            trig = pick_triggers(dk, cnt, cfg_eff, step_seed)
            # budget filter
            kept = []
            prompts, budgets = [], []
            for t in trig:
                i, k = t["row"], t["window"]
                if cfg["takeover_mode"] == "restart":
                    cut = 0  # full-demo mode: teacher prompt = bare question, budget = R
                else:
                    cut = (k - 1) * chunk if (t["mode"] == "random" or cfg["takeover_mode"] == "rewind") else k * chunk
                cut = min(cut, R)
                if R - cut < cfg["min_takeover_budget"]:
                    continue
                am = tb["attention_mask"][i]
                p_ids = tb["input_ids"][i, :P][am[:P].bool()].tolist()
                n_valid = int(rmask[i].sum().item())
                prompts.append(p_ids + tb["responses"][i, : min(cut, n_valid)].tolist())
                budgets.append(R - cut)
                kept.append(t)
            n_trig = len(kept)
            if n_trig == 0:
                tb["dtdd_origin_mask"] = torch.ones_like(tb["responses"], dtype=torch.float32)
                trainer._dtdd_stats(dk, cnt, kept, 0)
                return out1
            # round-robin across engines
            n_eng = len(trainer._teacher_pool)
            futs, order = [], []
            for e in range(n_eng):
                idx = list(range(e, n_trig, n_eng))
                if not idx:
                    continue
                futs.append(
                    trainer._teacher_pool[e].generate_batch.remote(
                        [prompts[j] for j in idx], [budgets[j] for j in idx]
                    )
                )
                order.extend(idx)
            flat = [x for res in ray.get(futs) for x in res]
            teacher_ids = [None] * n_trig
            for j, ids in zip(order, flat):
                teacher_ids[j] = ids
            origin = splice_batch(tb, kept, teacher_ids, chunk, pad_id, cfg["takeover_mode"])
            tb["dtdd_origin_mask"] = origin
            # refresh student logps / top-k on the spliced batch
            refreshed = actor_wg.compute_log_prob(batch)
            for key in ("old_log_probs", "student_top_k_log_probs", "student_top_k_ids", "entropys"):
                if key in refreshed.batch.keys():
                    tb[key] = refreshed.batch[key]
            out2 = rm_orig(batch)
            # sampled (2D) mode: BC override here (no distillation call follows)
            if "rm_scores" in out2.batch.keys() and out2.batch["rm_scores"].dim() == 2:
                sfx = (origin < 0.5) & (tb["response_mask"] > 0)
                out2.batch["rm_scores"][sfx] = cfg["bc_coef"]
            trainer._dtdd_stats(dk, cnt, kept, sum(len(t) for t in teacher_ids))
            return out2

        self.rm_wg.compute_rm_score = rm_wrapped

        if dist_orig is not None:

            def dist_wrapped(batch):
                out = dist_orig(batch)
                tb = batch.batch
                om = tb.get("dtdd_origin_mask", None)
                if om is not None and "rm_scores" in out.batch.keys() and out.batch["rm_scores"].dim() == 3:
                    sfx = (om < 0.5) & (tb["response_mask"] > 0)
                    rs = out.batch["rm_scores"]
                    sel = rs[sfx]
                    sel[:] = 0.0
                    sel[:, 0] = cfg["bc_coef"]
                    rs[sfx] = sel
                    ids = tb["student_top_k_ids"]
                    idsel = ids[sfx]
                    idsel[:, 0] = tb["responses"][sfx]
                    ids[sfx] = idsel
                return out

            self.actor_rollout_wg.compute_distillation_reward = dist_wrapped

    def _dtdd_stats(self, dk, cnt, kept, teacher_tokens):
        valid = cnt > 0
        vals = dk[valid]
        B = dk.shape[0]
        msg = {
            "dtdd/trigger_rate": len(kept) / max(B, 1),
            "dtdd/mean_switch_window": (np.mean([t["window"] for t in kept]) if kept else -1),
            "dtdd/teacher_tokens": teacher_tokens,
            "dtdd/dk_mean": float(vals.mean()) if vals.numel() else -1,
            "dtdd/dk_p90": float(vals.quantile(0.9)) if vals.numel() else -1,
        }
        if self.dtdd_cfg["dry_run"]:
            # Calibration aid: sequence-level first-passage trigger rate at candidate taus.
            minp = self.dtdd_cfg["min_prefix_turns"]
            eligible = dk.clone()
            eligible[cnt <= 0] = float("-inf")
            eligible[:, : minp - 1] = float("-inf")
            for tau in (0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6):
                rate = float((eligible > tau).any(dim=1).float().mean())
                msg[f"dtdd/would_trigger@tau{tau}"] = round(rate, 3)
            dump_dir = os.environ.get("DTDD_DK_DUMP")
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
                torch.save(
                    {"dk": dk.cpu(), "cnt": cnt.cpu()},
                    os.path.join(dump_dir, f"dk_step{getattr(self, 'global_steps', 0)}.pt"),
                )
        print(f"DTDD step {getattr(self, 'global_steps', '?')}: {msg}")
