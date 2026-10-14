# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DTDD trainer: ``RayPPOTrainer`` + frozen teacher pool + DTDD loss masks.

Extends the SOD trainer (YoungZ365/SOD@110c4b8) without touching verl core:

* ``init_workers()`` creates the named teacher server actors (``num_gpus=0``,
  made GPU-visible through a ``CUDA_VISIBLE_DEVICES`` runtime env, and pinned
  to the node of the actor-rollout workers), then wraps
  ``async_rollout_manager`` in a proxy that wakes and sleeps the teacher pool
  around each rollout and injects the per-sample ``dtdd_validate`` field;
* ``_apply_filter_groups()`` stashes the rollout-truth DTDD metrics before
  filtering, because the stock trainer filters before the regularizer hook;
* ``_apply_token_kl_regularizer()`` computes the prefix-restricted stepwise
  OPD advantage (pure distillation by default, ``grpo_prefix_coef=0``), the
  suffix SFT mask, the ``drop_failed_suffix`` option, the loss knobs handed
  to the actor through ``meta_info``, and the DTDD metrics.

Environment knobs read here:

``DTDD_TEACHER_NO_SLEEP=1``
    Keep the teacher engines resident instead of the per-rollout sleep/wake
    cycle. ``wake_up()`` re-maps physical pages through vLLM's CuMemAllocator
    and can fail with an out-of-memory error once the update phase has
    fragmented free memory. A resident teacher costs its full footprint during
    the update phase, so pair this with a lower student utilisation.
``DTDD_TEACHER_GPU_OFFSET=<k>``
    Put the teacher engines on GPUs the student does not use: allocate more
    devices than ``trainer.n_gpus_per_node`` and give the index of the first
    spare one. Sharing devices between the student and the frozen teacher has
    no working operating point -- a low teacher utilisation starves its KV
    cache (scoring became 97% of a step), while a high one makes the student
    engine fail to re-map its memory on wake. Every DTDD run that stayed
    healthy for hours gave the teacher its own devices; the paper's code arms
    use ``--gpus=8`` with ``n_gpus_per_node=4`` and an offset of 4.
"""

import logging
import os

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    compute_stepwise_opd_weights,
)

from .teacher_vllm_server import TeacherAsyncvLLMServer

logger = logging.getLogger(__file__)

TEACHER_SERVER_NAME = "dtdd_teacher_server"


class DTDDRolloutManagerProxy:
    """Wraps ``AgentLoopManager``: teacher wake/sleep + ``dtdd_validate``."""

    def __init__(self, inner, teacher_servers: list, config) -> None:
        self._inner = inner
        self._teacher_servers = teacher_servers
        self._config = config

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Run one rollout with the teacher pool awake around it.

        Args:
            prompts: The rollout batch; ``meta_info["validate"]`` marks a
                validation pass.

        Returns:
            The rollout output of the wrapped manager.
        """
        validate = bool(prompts.meta_info.get("validate", False))
        # Validation stays student-only: flag every sample so the agent loop
        # skips scoring/takeover and does not emit the origin mask.
        prompts.non_tensor_batch["dtdd_validate"] = np.array(
            [validate] * len(prompts), dtype=object
        )

        use_teacher = not validate
        no_sleep = os.environ.get("DTDD_TEACHER_NO_SLEEP", "0") == "1"
        if use_teacher and not no_sleep:
            ray.get([s.wake_up.remote() for s in self._teacher_servers])
        try:
            return self._inner.generate_sequences(prompts)
        finally:
            if use_teacher and not no_sleep:
                try:
                    ray.get([s.sleep.remote() for s in self._teacher_servers])
                except Exception as exc:
                    # A dead teacher engine (typically an out-of-memory error
                    # from the prompt_logprobs logits spike -- see
                    # dtdd.teacher.max_num_batched_tokens) must not mask the
                    # original rollout error with a sleep() failure.
                    logger.error(
                        f"DTDD teacher pool sleep failed -- engine likely "
                        f"dead (check teacher EngineCore OOM; "
                        f"dtdd.teacher.max_num_batched_tokens bounds the "
                        f"prompt_logprobs logits spike): {exc}"
                    )

    def wake_up(self) -> None:
        self._inner.wake_up()

    def sleep(self) -> None:
        self._inner.sleep()

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


class DTDDRayPPOTrainer(RayPPOTrainer):
    """``RayPPOTrainer`` with the frozen teacher pool and DTDD masks."""

    def _dtdd_enabled(self) -> bool:
        dtdd = self.config.get("dtdd", None)
        return bool(dtdd is not None and dtdd.enable)

    # ------------------------------------------------------------ workers --
    def init_workers(self) -> None:
        super().init_workers()
        if not self._dtdd_enabled():
            return

        dtdd = self.config.dtdd
        assert self.config.actor_rollout_ref.rollout.mode == "async", (
            "DTDD requires rollout.mode=async"
        )
        assert self.config.actor_rollout_ref.rollout.calculate_log_probs, (
            "DTDD requires actor_rollout_ref.rollout.calculate_log_probs=True"
        )
        token_kl_cfg = getattr(self.config.algorithm, "token_kl_reg", None)
        # `enable` is what makes the stock trainer call the regularizer hook
        # at all, so it stays mandatory. `stepwise_enable` is NOT required:
        # the override honours the flag (w_k = 1 when off), which is how the
        # code lane runs the same vanilla-OPD baseline as the other lanes.
        assert token_kl_cfg is not None and token_kl_cfg.enable, (
            "DTDD's prefix distillation rides on the token_kl_reg hook "
            "(only invoked when algorithm.token_kl_reg.enable): "
            "set +algorithm.token_kl_reg.enable=True"
        )
        ref_path = OmegaConf.select(
            self.config, "actor_rollout_ref.ref.model.path"
        )
        assert ref_path == dtdd.teacher.model_path, (
            f"Update-phase ref_log_prob must come from the SAME teacher as the "
            f"rollout trigger: actor_rollout_ref.ref.model.path={ref_path!r} "
            f"!= dtdd.teacher.model_path={dtdd.teacher.model_path!r} "
            "(without +actor_rollout_ref.ref.model.path the FSDP ref worker "
            "silently loads the student)"
        )

        self._init_dtdd_teacher_servers()
        self.async_rollout_manager = DTDDRolloutManagerProxy(
            self.async_rollout_manager, self.dtdd_teacher_servers, self.config
        )
        self._dtdd_prefilter_metrics = {}
        self._dtdd_high_trigger_streak = 0

    def _init_dtdd_teacher_servers(self) -> None:
        """Create, initialise and park the teacher engines."""
        teacher_cfg = self.config.dtdd.teacher
        assert self.config.trainer.nnodes == 1, (
            "DTDD teacher pool currently supports single-node runs"
        )
        n_gpus = self.config.trainer.n_gpus_per_node
        tp = int(teacher_cfg.tensor_model_parallel_size)
        assert n_gpus % tp == 0, (
            f"n_gpus_per_node={n_gpus} not divisible by teacher tp={tp}"
        )
        dp = n_gpus // tp

        # The FSDP workers own the Ray GPU resources; ask them for their node
        # and Ray-assigned CUDA_VISIBLE_DEVICES. The union is exactly this
        # job's physical GPU allocation, also under non-contiguous
        # allocations.
        workers = self.actor_rollout_wg.workers
        node_ids = ray.get(
            [
                w.__ray_call__.remote(
                    lambda self: ray.get_runtime_context().get_node_id()
                )
                for w in workers
            ]
        )
        cvds = ray.get(
            [
                w.__ray_call__.remote(
                    lambda self: os.environ.get("CUDA_VISIBLE_DEVICES", "")
                )
                for w in workers
            ]
        )
        gpu_ids = sorted(
            {g for cvd in cvds for g in cvd.split(",") if g}, key=int
        )
        assert len(gpu_ids) >= n_gpus, (
            f"collected GPU ids {gpu_ids} < trainer.n_gpus_per_node={n_gpus}"
        )
        node_id = node_ids[0]

        offset = os.environ.get("DTDD_TEACHER_GPU_OFFSET")
        if offset:
            offset = int(offset)
            # gpu_ids above comes from the STUDENT workers' visibility, so it
            # only ever contains the n_gpus_per_node devices they claimed; the
            # spare cards this offset points at are invisible there. Enumerate
            # the job's full allocation instead. Use container-local indices,
            # not the scheduler's physical device ids: inside the job every
            # allocated GPU is renumbered 0..N-1, so a physical id names
            # nothing. SLURM_JOB_GPUS is only used to learn HOW MANY devices
            # the job holds; outside Slurm the count falls back to
            # offset + dp * tp.
            allocation = (
                os.environ.get("SLURM_JOB_GPUS")
                or os.environ.get("SLURM_STEP_GPUS")
                or ""
            )
            n_total = len([g for g in allocation.split(",") if g]) or (
                offset + dp * tp
            )
            all_ids = [str(i) for i in range(n_total)]
            spare = all_ids[offset:]
            need = dp * tp
            assert len(spare) >= need, (
                f"DTDD_TEACHER_GPU_OFFSET={offset} leaves {len(spare)} spare "
                f"GPUs ({spare}) but the teacher pool needs {need} "
                f"(dp={dp} x tp={tp}); request --gpus={offset + need} with "
                f"trainer.n_gpus_per_node={n_gpus} (job holds {n_total} "
                f"GPUs; local ids {all_ids})"
            )
            gpu_ids = spare
            print(
                f"DTDD teacher pool on dedicated GPUs {gpu_ids[:need]} "
                f"(student keeps 0..{offset - 1})"
            )

        self.dtdd_teacher_servers = []
        for rank in range(dp):
            rank_cvd = ",".join(gpu_ids[rank * tp: (rank + 1) * tp])
            server = TeacherAsyncvLLMServer.options(
                name=f"{TEACHER_SERVER_NAME}_{rank}",
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False
                ),
                # num_gpus stays 0 (the FSDP worker group claims every GPU);
                # visibility comes from the runtime_env instead. Ray force-
                # sets CUDA_VISIBLE_DEVICES="" for 0-GPU actors, so that is
                # disabled (NOSET) and the devices are also passed through
                # DTDD_TEACHER_CVD, re-exported in init_engine.
                runtime_env={
                    "env_vars": {
                        "CUDA_VISIBLE_DEVICES": rank_cvd,
                        "DTDD_TEACHER_CVD": rank_cvd,
                        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                        "VLLM_USE_V1": "1",
                    }
                },
            ).remote(self.config, dp, rank, "dtdd_teacher")
            self.dtdd_teacher_servers.append(server)

        # Wait for FastAPI readiness, then init engines, then sleep at once
        # (level 1: weights -> CPU RAM). They wake per rollout via the proxy.
        servers = self.dtdd_teacher_servers
        ray.get([s.get_server_address.remote() for s in servers])
        ray.get([s.init_engine.remote() for s in servers])
        ray.get([s.sleep.remote() for s in servers])
        print(
            f"DTDD teacher pool ready: {dp} server(s), tp={tp}, "
            f"gpus={gpu_ids}, now sleeping at level 1"
        )

    # ------------------------------------------------------------ metrics --
    def _apply_filter_groups(self, batch: DataProto) -> DataProto:
        # Rollout-truth DTDD metrics must be aggregated after reward but
        # BEFORE filtering (the stock trainer filters before the hook).
        if self._dtdd_enabled():
            self._dtdd_prefilter_metrics = self._collect_dtdd_rollout_metrics(
                batch
            )
        return super()._apply_filter_groups(batch)

    def _collect_dtdd_rollout_metrics(self, batch: DataProto) -> dict:
        """Aggregate the per-episode ``dtdd_*`` telemetry into step metrics.

        Args:
            batch: The post-reward, pre-filter rollout batch.

        Returns:
            A ``dtdd/*`` metric dict; empty when no episode carried telemetry.
        """
        extra_infos = batch.non_tensor_batch.get("extra_info", None)
        if extra_infos is None:
            return {}
        infos = [
            info
            for info in extra_infos
            if isinstance(info, dict) and "dtdd_triggered" in info
        ]
        if not infos:
            return {}

        n = len(infos)
        triggered = np.array(
            [info["dtdd_triggered"] for info in infos], dtype=np.float64
        )
        modes = [info.get("dtdd_trigger_mode") for info in infos]
        switch_turns = np.array(
            [info.get("dtdd_switch_turn", -1) for info in infos],
            dtype=np.float64,
        )
        teacher_tokens = np.array(
            [info.get("dtdd_teacher_tokens", 0) for info in infos],
            dtype=np.float64,
        )
        empty_prefix = np.array(
            [info.get("dtdd_empty_prefix", 0) for info in infos],
            dtype=np.float64,
        )
        would_trigger = np.array(
            [info.get("dtdd_would_trigger_turn", -1) for info in infos],
            dtype=np.float64,
        )
        score_failures = float(
            sum(info.get("dtdd_score_failures", 0) for info in infos)
        )
        score_ms = np.array(
            [info.get("dtdd_score_ms_total", 0.0) for info in infos],
            dtype=np.float64,
        )
        all_dk = np.array(
            [v for info in infos for v in info.get("dtdd_dk_values", [])],
            dtype=np.float64,
        )

        response_tokens = float(batch.batch["response_mask"].sum().item())
        metrics = {
            "dtdd/trigger_rate": float(triggered.mean()),
            "dtdd/teacher_token_frac": float(
                teacher_tokens.sum() / max(response_tokens, 1.0)
            ),
            "dtdd/empty_prefix_rate": float(empty_prefix.mean()),
            "dtdd/would_trigger_rate": float((would_trigger > 0).mean()),
            "dtdd/score_failures": score_failures,
            "dtdd/score_ms_mean": float(score_ms.mean()),
        }
        for mode in ("divergence", "random", "random_skipped"):
            sel = np.array([m == mode for m in modes], dtype=bool)
            if sel.any():
                metrics[f"dtdd/trigger_rate_{mode}"] = float(
                    triggered[sel].mean()
                )
                metrics[f"dtdd/mode_frac_{mode}"] = float(sel.mean())
        if (triggered > 0).any():
            metrics["dtdd/mean_switch_turn"] = float(
                switch_turns[triggered > 0].mean()
            )
        if len(all_dk):
            for q in (50, 90):
                metrics[f"dtdd/dk_p{q}"] = float(np.percentile(all_dk, q))
            metrics["dtdd/dk_mean"] = float(all_dk.mean())
            metrics["dtdd/dk_frac_gt_tau"] = float(
                (all_dk > float(self.config.dtdd.tau)).mean()
            )

        # Teacher recovery success rate: reward over triggered episodes.
        if "token_level_scores" in batch.batch and (triggered > 0).any():
            seq_reward = batch.batch["token_level_scores"].sum(-1).cpu().numpy()
            # extra_info rows align 1:1 with batch rows (built per sample).
            if len(seq_reward) == n:
                thr = float(self.config.dtdd.suffix_success_threshold)
                metrics["dtdd/teacher_recovery_success_rate"] = float(
                    (seq_reward[triggered > 0] > thr).mean()
                )

        # Collapse rail: warn on sustained near-total triggering.
        if metrics["dtdd/trigger_rate"] > 0.9:
            self._dtdd_high_trigger_streak += 1
            if self._dtdd_high_trigger_streak >= 3:
                logger.warning(
                    f"DTDD trigger_rate > 0.9 for "
                    f"{self._dtdd_high_trigger_streak} consecutive steps -- "
                    "tau too low or student collapsing; check dtdd/dk_* metrics"
                )
        else:
            self._dtdd_high_trigger_streak = 0
        metrics["dtdd/high_trigger_streak"] = float(
            self._dtdd_high_trigger_streak
        )
        return metrics

    # --------------------------------------------------------------- loss --
    def _apply_token_kl_regularizer(self, batch: DataProto, cfg) -> DataProto:
        """Prefix-restricted stepwise OPD + suffix SFT masks.

        Delegates to the stock hook when DTDD is off or the batch carries no
        origin mask.

        Args:
            batch: The training batch after advantage computation.
            cfg: ``algorithm.token_kl_reg``.

        Returns:
            The batch with ``advantages`` replaced by the DTDD advantage and
            ``prefix_mask`` / ``suffix_sft_mask`` ``(B, L)`` added.
        """
        if not self._dtdd_enabled() or "teacher_kl_mask" not in batch.batch:
            return super()._apply_token_kl_regularizer(batch, cfg)
        if (
            cfg is None
            or not getattr(cfg, "enable", False)
            or "advantages" not in batch.batch
        ):
            return batch
        if (
            "ref_log_prob" not in batch.batch
            or "old_log_probs" not in batch.batch
        ):
            return batch

        dtdd = self.config.dtdd
        response_mask = batch.batch["response_mask"].float()
        # 1 = student-origin assistant token.
        origin = batch.batch["teacher_kl_mask"].float()
        prefix_mask = response_mask * origin
        suffix_mask = response_mask * (1.0 - origin)

        # ---- prefix: stepwise-weighted OPD on student-origin steps only.
        # _extract_step_boundaries operates on whatever mask it is given, so
        # passing prefix_mask means teacher-suffix runs simply are not steps.
        epsilon = float(getattr(cfg, "stepwise_epsilon", 1e-6))
        delta = float(getattr(cfg, "stepwise_delta", 0.5))
        opd_coef = float(getattr(cfg, "stepwise_opd_coef", 1.0))

        raw_local_adv = (
            batch.batch["ref_log_prob"] - batch.batch["old_log_probs"]
        ) * prefix_mask
        # stepwise_enable mirrors the stock path: w_k weighting is SOD's
        # contribution, not vanilla OPD. The override MUST honour the same
        # flag, otherwise STEPWISE_ENABLE=False would silently leave DTDD
        # weighted while the OPD baseline runs unweighted -- a confound in
        # the very comparison the flag exists to keep clean.
        if bool(getattr(cfg, "stepwise_enable", False)):
            stepwise_weights, stepwise_log_info = compute_stepwise_opd_weights(
                old_log_probs=batch.batch["old_log_probs"],
                ref_log_prob=batch.batch["ref_log_prob"],
                response_mask=prefix_mask,
                epsilon=epsilon,
                delta=delta,
            )
            stepwise_weights = stepwise_weights.to(raw_local_adv.device)
            stepwise_summary = self._write_stepwise_log(
                stepwise_log_info, epsilon, delta, opd_coef
            )
        else:
            stepwise_weights = torch.ones_like(raw_local_adv)
            stepwise_summary = {"opd/stepwise_weight_mean": 1.0}

        weighted_opd = opd_coef * stepwise_weights * raw_local_adv

        # ---- GRPO term: 0 in pure-distillation mode (the default).
        grpo_coef = float(dtdd.grpo_prefix_coef)
        grpo_adv = batch.batch["advantages"]
        if grpo_adv.dim() == 1:
            grpo_adv = grpo_adv.unsqueeze(1)
        # Zero on the suffix by construction.
        total_advantage = grpo_coef * grpo_adv * prefix_mask + weighted_opd
        batch.batch["advantages"] = total_advantage

        # ---- suffix SFT mask (+ optional drop of failed teacher recoveries).
        if dtdd.drop_failed_suffix and "token_level_scores" in batch.batch:
            seq_reward = batch.batch["token_level_scores"].sum(-1)
            keep = (
                (seq_reward > float(dtdd.suffix_success_threshold))
                .float()
                .unsqueeze(1)
            )
            suffix_mask = suffix_mask * keep

        batch.batch["prefix_mask"] = prefix_mask
        batch.batch["suffix_sft_mask"] = suffix_mask
        batch.meta_info["dtdd_suffix_sft_coef"] = float(dtdd.suffix_sft_coef)
        batch.meta_info["dtdd_suffix_loss_type"] = str(dtdd.suffix_loss_type)

        # ---- metrics: post-filter mask/loss metrics + the stashed
        # pre-filter rollout truth.
        stepwise_summary = dict(stepwise_summary)
        stepwise_summary.update(self._dtdd_prefilter_metrics)
        self._dtdd_prefilter_metrics = {}
        n_resp = response_mask.sum().clamp_min(1.0)
        stepwise_summary["dtdd/suffix_token_frac_train"] = float(
            suffix_mask.sum().item() / n_resp.item()
        )
        stepwise_summary["dtdd/prefix_token_frac_train"] = float(
            prefix_mask.sum().item() / n_resp.item()
        )
        # Corrected version of training/rollout_probs_diff: the stock debug
        # metric is response_mask-only and thus polluted by teacher-policy
        # suffix logprobs on DTDD runs; this one is prefix-masked.
        if "rollout_log_probs" in batch.batch:
            with torch.no_grad():
                diff = (
                    batch.batch["old_log_probs"]
                    - batch.batch["rollout_log_probs"]
                ).abs() * prefix_mask
                stepwise_summary["dtdd/rollout_probs_diff_prefix"] = float(
                    (diff.sum() / prefix_mask.sum().clamp_min(1.0)).item()
                )

        batch.meta_info.setdefault("token_kl_reg", {})
        batch.meta_info["token_kl_reg"].update(
            {
                "stepwise_weights": stepwise_weights.detach().clone(),
                "local_adv": raw_local_adv.detach().clone(),
                "stepwise_summary": stepwise_summary,
            }
        )
        return batch
