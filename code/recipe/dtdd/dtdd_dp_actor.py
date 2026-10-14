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
"""DTDD actor: prefix-masked PPO plus a suffix SFT (CE or soft-KL) loss.

``update_policy`` is a pinned copy of ``DataParallelPPOActor.update_policy``
(``verl/workers/actor/dp_actor.py`` at YoungZ365/SOD@110c4b8) with these
DTDD deltas:

* ``select_keys`` gains ``prefix_mask`` / ``suffix_sft_mask`` (and
  ``ref_log_prob`` for the suffix-KL ablation) so the masks survive
  mini-batching;
* the PPO and entropy terms are aggregated over ``prefix_mask`` when it is
  present, so teacher-origin tokens never enter the ratio, the clipping or
  the entropy bonus;
* a suffix loss is added over ``suffix_sft_mask``: ``-log_prob`` (``ce``) or
  the low-variance KL to the teacher (``kl``), scaled by
  ``data.meta_info["dtdd_suffix_sft_coef"]``;
* zero-token guards on both masks. A micro-batch can be all-suffix (a rewind
  at turn 1 leaves an empty student prefix) or all-prefix (no hybrid
  episodes); masked aggregation over an empty mask would yield NaN.

When the DTDD masks are absent from the batch (``dtdd.enable=false``, or the
trainer hook did not run) the method delegates to the base actor, so the
behaviour is byte-for-byte the stock one.

Tensor shapes below use ``B`` for the micro-batch size and ``L`` for the
response length.
"""

import torch

from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor


class DTDDDataParallelPPOActor(DataParallelPPOActor):
    """``DataParallelPPOActor`` whose policy update honours DTDD masks."""

    def update_policy(self, data: DataProto) -> dict:
        """Run one policy update, splitting the loss by token origin.

        Args:
            data: The training batch. With DTDD active it carries
                ``prefix_mask`` and ``suffix_sft_mask`` ``(B, L)`` tensors and
                the ``dtdd_suffix_*`` entries in ``meta_info``.

        Returns:
            A dict of per-micro-batch metric lists, as the base actor returns.
        """
        if "suffix_sft_mask" not in data.batch.keys():
            # Non-DTDD batch (dtdd.enable=false or hook not applied).
            return super().update_policy(data)

        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        suffix_sft_coef = float(data.meta_info.get("dtdd_suffix_sft_coef", 0.0))
        suffix_loss_type = str(
            data.meta_info.get("dtdd_suffix_loss_type", "ce")
        )
        assert suffix_loss_type in ("ce", "kl"), (
            f"unknown dtdd suffix_loss_type: {suffix_loss_type}"
        )

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            # DTDD: masks must be selected or they vanish before minibatching.
            "prefix_mask",
            "suffix_sft_mask",
        ]
        if self.config.use_kl_loss or suffix_loss_type == "kl":
            select_keys.append("ref_log_prob")
        if self.config.tis_imp_ratio_cap > 0:
            assert "rollout_log_probs" in data.batch.keys(), (
                "Truncated Importance Sampling (TIS) requires to configure "
                "`actor_rollout_ref.rollout.calculate_log_probs=True` "
                "and is not currently supported in Server mode (agent loop)."
            )
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = (
            "multi_modal_inputs" in data.non_tensor_batch.keys()
        )
        non_tensor_select_keys = (
            ["multi_modal_inputs"] if has_multi_modal_inputs else []
        )

        data = data.select(
            batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys
        )

        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = (
                        self.config.ppo_max_token_len_per_gpu
                        * self.ulysses_sequence_parallel_size
                    )
                    micro_batches, _ = prepare_dynamic_batch(
                        mini_batch, max_token_len=max_token_len
                    )
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size
                        // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(
                        self.config.ppo_micro_batch_size_per_gpu
                    )

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {
                        **micro_batch.batch, **micro_batch.non_tensor_batch
                    }
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    rollout_log_probs = (
                        model_inputs["rollout_log_probs"]
                        if self.config.tis_imp_ratio_cap > 0
                        else None
                    )
                    advantages = model_inputs["advantages"]

                    # DTDD: PPO/entropy over student-origin tokens only.
                    prefix_mask = model_inputs["prefix_mask"]
                    suffix_mask = model_inputs["suffix_sft_mask"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = (
                            response_mask.shape[0]
                            / self.config.ppo_mini_batch_size
                        )
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )

                    if on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs["old_log_probs"]

                    # Guard: a micro-batch can be all-suffix (rewind at turn 1)
                    # or all-prefix (no hybrid episodes); masked aggregation
                    # over an empty mask would produce NaN.
                    has_prefix = bool(prefix_mask.sum() > 0)
                    has_suffix = bool(suffix_mask.sum() > 0)

                    if has_prefix:
                        loss_mode = self.config.policy_loss.get(
                            "loss_mode", "vanilla"
                        )
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = (
                            policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=prefix_mask,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_log_probs=rollout_log_probs,
                            )
                        )
                    else:
                        # A zero that keeps a valid grad graph for backward().
                        zero = log_prob.sum() * 0.0
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = (
                            zero, zero.detach(), zero.detach(), zero.detach()
                        )

                    if entropy_coeff != 0 and has_prefix:
                        entropy_loss = agg_loss(
                            loss_mat=entropy,
                            loss_mask=prefix_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss and has_prefix:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob,
                            ref_logprob=ref_log_prob,
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl_loss = agg_loss(
                            loss_mat=kld,
                            loss_mask=prefix_mask,
                            loss_agg_mode=loss_agg_mode,
                        )

                        policy_loss = (
                            policy_loss + kl_loss * self.config.kl_loss_coef
                        )
                        micro_batch_metrics["actor/kl_loss"] = (
                            kl_loss.detach().item() * loss_scale_factor
                        )
                        micro_batch_metrics["actor/kl_coef"] = (
                            self.config.kl_loss_coef
                        )

                    # DTDD suffix SFT: learn the teacher's recovery tokens.
                    if suffix_sft_coef > 0 and has_suffix:
                        if suffix_loss_type == "ce":
                            # NLL of the teacher-generated tokens.
                            sft_mat = -log_prob
                        else:
                            # "kl": soft distillation toward the teacher.
                            sft_mat = kl_penalty(
                                logprob=log_prob,
                                ref_logprob=model_inputs["ref_log_prob"],
                                kl_penalty="low_var_kl",
                            )
                        sft_loss = agg_loss(
                            loss_mat=sft_mat,
                            loss_mask=suffix_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        policy_loss = policy_loss + suffix_sft_coef * sft_loss
                        micro_batch_metrics["actor/dtdd_sft_loss"] = (
                            sft_loss.detach().item() * loss_scale_factor
                        )
                        micro_batch_metrics["actor/dtdd_suffix_token_frac"] = (
                            (
                                suffix_mask.sum()
                                / response_mask.sum().clamp_min(1.0)
                            )
                            .detach()
                            .item()
                        )

                    # Both branches scale identically; kept as two branches to
                    # stay a line-for-line copy of the upstream method.
                    if self.config.use_dynamic_bsz:
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": (
                                pg_loss.detach().item() * loss_scale_factor
                            ),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": (
                                pg_clipfrac_lower.detach().item()
                            ),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {
                    "actor/grad_norm": grad_norm.detach().item()
                }
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
