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
"""Standalone frozen teacher vLLM server for DTDD.

Unlike ``AsyncvLLMServer`` (``verl/workers/rollout/vllm_rollout/
vllm_async_server.py`` at YoungZ365/SOD@110c4b8), which hosts its engine
inside the hybrid FSDP rollout workers through an external executor and is
weight-synced every step, this server:

* loads real teacher weights once (``load_format="auto"``) and never syncs;
* runs vLLM's default multiprocess executor inside its own Ray actor
  (``num_gpus=0``, made GPU-visible through the runtime env);
* sleeps at level 1 only (weights to CPU RAM; level 2 would discard them);
* additionally exposes ``score()`` for ``prompt_logprobs``-based divergence
  scoring.

``generate()``, ``wake_up()`` and ``sleep()`` are pinned copies of the
upstream methods with the teacher-specific deltas noted inline.

Environment knobs read in this actor process (a Ray ``runtime_env`` can
shadow the job environment, so the startup banner prints what the actor
actually sees):

``DTDD_TEACHER_CVD``
    The devices this engine may use; set by the trainer. Ray force-sets
    ``CUDA_VISIBLE_DEVICES=""`` for 0-GPU actors after the runtime env is
    applied, so the value is re-exported here before any CUDA call.
``DTDD_TEACHER_ATTN_BACKEND``
    Override the teacher engine's attention backend (for example
    ``XFORMERS``). Scoped to this process; the student engine keeps its
    default. Added after a Qwen2.5-7B teacher hit an illegal memory access
    inside ``flash_attn_varlen_func`` on A40s while a Qwen3-4B teacher on
    the same code path was fine.
``DTDD_TEACHER_SERIALIZE=1``
    Mutex ``score()`` and ``generate()`` entirely. Resident (no-sleep)
    engines died with an illegal memory access under concurrent
    ``prompt_logprobs`` scoring and generation on both the flash-attn and
    Triton backends and with two different teachers, while sleep-mode
    engines (reset every step) survived the same traffic; a cross-request
    state race in vLLM V1 ``AsyncLLM`` is the remaining suspect. A full mutex
    is safe but admits one request at a time, so vLLM can never batch.
``DTDD_TEACHER_OPGATE=1``
    The middle ground: any number of same-kind operations may run together;
    only scoring and generation are kept from being in flight simultaneously.
    ``prompt_logprobs`` and sampling never share an engine step, while
    batching inside each kind is preserved.
``DTDD_TEACHER_VOCAB_CAP``
    See ``dtdd_agent_loop.py``; here it selects the token-id sanitising
    described in ``init_engine``.
"""

import asyncio
import contextlib
import logging
import os
from typing import Any, Optional
from uuid import uuid4

import ray
from omegaconf import DictConfig
from starlette.requests import Request
from starlette.responses import JSONResponse
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput
from vllm.v1.engine.async_llm import AsyncLLM

from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.workers.rollout.async_server import AsyncServerBase, TokenOutput
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    _ensure_transformers_tokenizer_compat,
    _qwen2_5_vl_dedup_image_tokens,
)

logger = logging.getLogger(__file__)


@ray.remote(num_cpus=1)
class TeacherAsyncvLLMServer(AsyncServerBase):
    """Frozen teacher engine: takeover generation + prompt_logprobs scoring."""

    def __init__(
        self,
        config: DictConfig,
        vllm_dp_size: int,
        vllm_dp_rank: int,
        wg_prefix: str,
    ) -> None:
        super().__init__()
        self.full_config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.teacher_cfg = config.dtdd.teacher
        self.vllm_dp_size = vllm_dp_size
        self.vllm_dp_rank = vllm_dp_rank
        self.wg_prefix = wg_prefix
        self.engine: AsyncLLM = None
        self._sleeping = False
        serialize = os.environ.get("DTDD_TEACHER_SERIALIZE", "0") == "1"
        self._op_lock = asyncio.Lock() if serialize else None
        self._gate_on = os.environ.get("DTDD_TEACHER_OPGATE", "0") == "1"
        self._gate_kind = None  # "gen" | "score" | None
        self._gate_n = 0  # in-flight ops of that kind
        self._gate_cv = asyncio.Condition()

    @contextlib.asynccontextmanager
    async def _op_gate(self, kind: str):
        """Admit ``kind`` only when no op of the other kind is in flight."""
        if not self._gate_on:
            yield
            return
        async with self._gate_cv:
            while self._gate_kind is not None and self._gate_kind != kind:
                await self._gate_cv.wait()
            self._gate_kind = kind
            self._gate_n += 1
        try:
            yield
        finally:
            async with self._gate_cv:
                self._gate_n -= 1
                if self._gate_n == 0:
                    self._gate_kind = None
                    self._gate_cv.notify_all()

    async def init_engine(self) -> None:
        """Build the vLLM engine on the devices the trainer assigned."""
        teacher_cvd = os.environ.get("DTDD_TEACHER_CVD")
        if teacher_cvd:
            os.environ["CUDA_VISIBLE_DEVICES"] = teacher_cvd

        attn_backend = os.environ.get("DTDD_TEACHER_ATTN_BACKEND")
        if attn_backend:
            os.environ["VLLM_ATTENTION_BACKEND"] = attn_backend
            print(
                f"[teacher_vllm_server] attention backend override: "
                f"{attn_backend}"
            )

        # Imported after the CUDA_VISIBLE_DEVICES re-export above so the
        # vLLM mp-executor subprocesses inherit the right visibility.
        import torch

        teacher_cfg = self.teacher_cfg
        rollout_cfg = self.rollout_config

        tp = int(teacher_cfg.tensor_model_parallel_size)
        visible = torch.cuda.device_count()
        assert visible >= tp, (
            f"Teacher server sees {visible} GPUs (CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}) but needs "
            f"tensor_model_parallel_size={tp}"
        )

        model_path = teacher_cfg.model_path
        assert model_path, "dtdd.teacher.model_path must be set"
        local_path = copy_to_local(model_path)
        trust_remote_code = self.full_config.actor_rollout_ref.model.get(
            "trust_remote_code", False
        )

        # Intra-family distillation: teacher and student tokenizers must
        # match, or the teacher's vocabulary must be a prefix of the
        # student's and DTDD_TEACHER_VOCAB_CAP set (a Qwen2.5 teacher with a
        # Qwen3 student: same BPE, the student adds a few specials). The cap
        # is safe because (a) takeover generation is truncated at the first
        # id >= cap, all valid student ids; and (b) scoring student tokens
        # works since the teacher EMBEDDING (config vocab_size) covers the
        # student id range -- only the student's extra specials map to
        # untrained rows.
        student_local = copy_to_local(
            self.full_config.actor_rollout_ref.model.path
        )
        student_tok = hf_tokenizer(student_local, trust_remote_code=True)
        teacher_tok = hf_tokenizer(local_path, trust_remote_code=True)
        teacher_vocab = len(teacher_tok.get_vocab())
        student_vocab = len(student_tok.get_vocab())
        same_eos = teacher_tok.eos_token_id == student_tok.eos_token_id
        vocab_cap = int(os.environ.get("DTDD_TEACHER_VOCAB_CAP", "0"))
        self._sanitize_above = None
        self._sanitize_eos = teacher_tok.eos_token_id
        if teacher_vocab == student_vocab and same_eos:
            pass
        elif (
            vocab_cap > 0
            and same_eos
            and teacher_vocab <= student_vocab
            and vocab_cap <= teacher_vocab
        ):
            # Student ids >= teacher tokenizer length (e.g. Qwen3 <think>
            # specials) must be sanitised before vLLM's tokenizer-length
            # validation sees them.
            self._sanitize_above = teacher_vocab
            print(
                f"[teacher_vllm_server] WARNING: vocab mismatch tolerated "
                f"under DTDD_TEACHER_VOCAB_CAP={vocab_cap}: teacher "
                f"{teacher_vocab} vs student {student_vocab}, shared eos."
            )
        else:
            raise AssertionError(
                f"DTDD requires intra-family distillation with identical "
                f"vocabularies: teacher vocab={teacher_vocab}/eos="
                f"{teacher_tok.eos_token_id} vs student vocab={student_vocab}"
                f"/eos={student_tok.eos_token_id} (set "
                f"DTDD_TEACHER_VOCAB_CAP<=teacher vocab to tolerate a subset "
                f"teacher)"
            )

        # Mandatory 1 generated token for prompt_logprobs scoring, so leave
        # headroom.
        self.max_model_len = (
            int(rollout_cfg.prompt_length)
            + int(rollout_cfg.response_length)
            + 8
        )

        override_generation_config = dict(
            n=1,
            logprobs=0,
            repetition_penalty=1.0,
            max_new_tokens=int(rollout_cfg.response_length),
            temperature=float(teacher_cfg.temperature),
            top_p=float(teacher_cfg.top_p),
        )

        engine_args = AsyncEngineArgs(
            model=local_path,
            # Forced: the teacher must be able to sleep at level 1.
            enable_sleep_mode=True,
            override_generation_config=override_generation_config,
            tensor_parallel_size=tp,
            # Default mp executor, NOT the FSDP-worker executor.
            distributed_executor_backend=None,
            dtype=teacher_cfg.dtype,
            enforce_eager=teacher_cfg.enforce_eager,
            gpu_memory_utilization=float(teacher_cfg.gpu_memory_utilization),
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=self.max_model_len,
            max_num_seqs=int(teacher_cfg.max_num_seqs),
            # Real teacher weights; never synced.
            load_format="auto",
            disable_log_stats=rollout_cfg.disable_log_stats,
            max_num_batched_tokens=int(teacher_cfg.max_num_batched_tokens),
            enable_chunked_prefill=rollout_cfg.enable_chunked_prefill,
            # vLLM V1 rejects prompt_logprobs with prefix caching.
            enable_prefix_caching=bool(teacher_cfg.enable_prefix_caching),
            trust_remote_code=trust_remote_code,
            seed=0,
        )

        _ensure_transformers_tokenizer_compat()
        vllm_config = engine_args.create_engine_config()
        self.engine = AsyncLLM.from_vllm_config(vllm_config)
        self.processor = hf_processor(
            local_path, trust_remote_code=trust_remote_code, use_fast=True
        )
        print(
            f"DTDD teacher engine ready: {model_path} tp={tp} "
            f"gpu_mem_util={teacher_cfg.gpu_memory_utilization} "
            f"max_model_len={self.max_model_len} "
            f"| OPGATE={'ON' if self._gate_on else 'off'} "
            f"SERIALIZE={'ON' if self._op_lock is not None else 'off'} "
            f"NO_SLEEP={os.environ.get('DTDD_TEACHER_NO_SLEEP', '0')} "
            f"max_num_seqs={teacher_cfg.max_num_seqs}"
        )

    async def chat_completion(self, raw_request: Request) -> JSONResponse:
        return JSONResponse(
            content={
                "error": "chat_completion is not supported on the DTDD "
                "teacher server"
            },
            status_code=501,
        )

    # Pinned copy of AsyncvLLMServer.generate. Teacher delta: an explicit
    # "max_tokens" in sampling_params caps the engine-computed budget instead
    # of colliding with it.
    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate a teacher continuation of ``prompt_ids``.

        Args:
            prompt_ids: The full token sequence so far.
            sampling_params: vLLM sampling parameters as a plain dict.
            request_id: The episode's request id.
            image_data: Optional images for multimodal prompts.

        Returns:
            The generated tokens and, if requested, their logprobs.
        """
        if self._op_lock is not None:
            async with self._op_lock:
                return await self._generate_inner(
                    prompt_ids, sampling_params, request_id, image_data
                )
        async with self._op_gate("gen"):
            return await self._generate_inner(
                prompt_ids, sampling_params, request_id, image_data
            )

    async def _generate_inner(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        max_tokens = self.max_model_len - len(prompt_ids)
        sampling_params = dict(sampling_params)
        requested_max = sampling_params.pop("max_tokens", None)
        if requested_max is not None:
            max_tokens = min(max_tokens, int(requested_max))
        sampling_params["logprobs"] = (
            0 if sampling_params.pop("logprobs", False) else None
        )
        sampling_params = SamplingParams(
            max_tokens=max_tokens, **sampling_params
        )
        if self._sanitize_above is not None:
            prompt_ids = [
                i if i < self._sanitize_above else self._sanitize_eos
                for i in prompt_ids
            ]
        prompt_ids = _qwen2_5_vl_dedup_image_tokens(prompt_ids, self.processor)
        prompt = TokensPrompt(
            prompt_token_ids=prompt_ids,
            multi_modal_data={"image": image_data} if image_data else None,
        )
        generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        )

        final_res: Optional[RequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        token_ids = final_res.outputs[0].token_ids
        log_probs = None
        if sampling_params.logprobs is not None:
            log_probs = [
                logprobs[token_ids[i]].logprob
                for i, logprobs in enumerate(final_res.outputs[0].logprobs)
            ]
        if self._sanitize_above is not None:
            # Cap enforcement lives here rather than in the sampler (a
            # 151k-element allowed_token_ids list corrupted GPU memory at
            # takeover scale). Ids >= cap are untrained teacher rows with
            # essentially zero sampling mass; if one ever appears, truncate
            # the suffix there rather than splicing an invalid id.
            for position, token in enumerate(token_ids):
                if token >= self._sanitize_above:
                    logger.warning(
                        f"teacher sampled OOV id {token} at pos {position}; "
                        f"truncating suffix"
                    )
                    token_ids = token_ids[:position]
                    if log_probs is not None:
                        log_probs = log_probs[:position]
                    break
        return TokenOutput(token_ids=token_ids, log_probs=log_probs)

    async def score(
        self, prompt_ids: list[int], request_id: str, num_tokens: int
    ) -> TokenOutput:
        """Teacher logprobs of the last ``num_tokens`` of ``prompt_ids``.

        ``prompt_logprobs[i]`` is the logprob of prompt token ``i`` given the
        tokens before it (entry 0 is ``None``), so slicing the tail gives
        exactly the freshly generated turn's teacher logprobs. vLLM requires
        ``max_tokens >= 1`` for the call.

        Args:
            prompt_ids: The full token sequence so far.
            request_id: The episode's request id.
            num_tokens: How many trailing tokens to score.

        Returns:
            A ``TokenOutput`` with empty ``token_ids`` and ``num_tokens``
            logprobs.
        """
        if self._op_lock is not None:
            async with self._op_lock:
                return await self._score_inner(
                    prompt_ids, request_id, num_tokens
                )
        async with self._op_gate("score"):
            return await self._score_inner(
                prompt_ids, request_id, num_tokens
            )

    async def _score_inner(
        self, prompt_ids: list[int], request_id: str, num_tokens: int
    ) -> TokenOutput:
        if self._sanitize_above is not None:
            prompt_ids = [
                i if i < self._sanitize_above else self._sanitize_eos
                for i in prompt_ids
            ]
        assert 0 < num_tokens < len(prompt_ids), (
            f"score() needs 0 < num_tokens({num_tokens}) < "
            f"len(prompt_ids)({len(prompt_ids)}) (entry 0 of prompt_logprobs "
            "is None, so the scored span must not start at position 0)"
        )
        sampling_params = SamplingParams(
            max_tokens=1, prompt_logprobs=0, temperature=1.0, detokenize=False
        )
        prompt = TokensPrompt(prompt_token_ids=prompt_ids)
        # Unique id per score call: vLLM rejects duplicate in-flight ids.
        score_request_id = f"{request_id}-score-{uuid4().hex[:8]}"
        generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=score_request_id,
        )

        final_res: Optional[RequestOutput] = None
        async for output in generator:
            final_res = output
        assert (
            final_res is not None and final_res.prompt_logprobs is not None
        ), (
            "vLLM returned no prompt_logprobs -- V1 + prompt_logprobs "
            "unsupported in this build? (known incompatibility with "
            "enable_prefix_caching=True)"
        )

        tail = final_res.prompt_logprobs[-num_tokens:]
        log_probs = []
        for tok, lp_dict in zip(prompt_ids[-num_tokens:], tail):
            if lp_dict is not None and tok in lp_dict:
                log_probs.append(float(lp_dict[tok].logprob))
            else:
                # The None at position 0 is excluded by the assert above;
                # treat any residual miss as neutral.
                log_probs.append(0.0)
        return TokenOutput(token_ids=[], log_probs=log_probs)

    async def wake_up(self) -> None:
        if not self._sleeping:
            return
        await self.engine.wake_up()
        self._sleeping = False

    async def sleep(self) -> None:
        if self._sleeping:
            return
        # Level 1: weights -> CPU RAM, KV discarded. NEVER level 2
        # (VLLM_SLEEP_LEVEL defaults to 2 on vllm>=0.8.5): the teacher has no
        # weight-sync path to restore from.
        await self.engine.reset_prefix_cache()
        await self.engine.sleep(level=int(self.teacher_cfg.sleep_level))
        self._sleeping = True
