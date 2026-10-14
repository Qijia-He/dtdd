"""Frozen-teacher generation pool for DTDD on the Thinking-Space/Rethinking-OPD verl v0.7 fork.

Minimal sibling of SOD recipe/dtdd/teacher_vllm_server.py: this fork has no
AsyncvLLMServer, and DTDD's fit-loop stage is synchronous, so we wrap the
plain sync vllm.LLM engine in a Ray actor. Scoring is NOT done here — the
fp32 FSDP reward-model worker already produces exact teacher logprobs; this
pool only writes takeover continuations.

GPU placement mirrors the SOD trick: num_gpus=0 actor, real visibility via
DTDD_TEACHER_CVD + RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES in the
runtime_env, so vllm's mp executor subprocesses inherit the right devices
while Ray's scheduler never reserves them (they are shared with FSDP/vllm
colocated workers that own the allocation).
"""

import os

import ray


@ray.remote(num_cpus=1)
class TeacherGenPool:
    def __init__(self, model_path: str, teacher_cfg: dict, max_model_len: int):
        cvd = os.environ.get("DTDD_TEACHER_CVD")
        if cvd:
            os.environ["CUDA_VISIBLE_DEVICES"] = cvd
        from vllm import LLM

        self.cfg = dict(teacher_cfg)
        self.llm = LLM(
            model=model_path,
            dtype=self.cfg.get("dtype", "bfloat16"),
            tensor_parallel_size=int(self.cfg.get("tensor_parallel_size", 1)),
            gpu_memory_utilization=float(self.cfg.get("gpu_memory_utilization", 0.15)),
            max_model_len=max_model_len,
            enforce_eager=bool(self.cfg.get("enforce_eager", True)),
            enable_prefix_caching=False,
            disable_custom_all_reduce=True,
            trust_remote_code=True,
            seed=0,
        )
        # Teacher tokens are spliced into the student sequence, so when the
        # teacher embedding is padded larger than the student's (e.g. R1-7B
        # 152064 vs R1-1.5B 151936; same tokenizer, extra rows are untrained
        # padding), sampling must be capped to ids the student can embed.
        # vLLM validates allowed_token_ids against len(tokenizer) (real tokens,
        # < embedding size), so cap to the tighter of the two — ids in
        # [len(tokenizer), embedding) are padding on both sides anyway.
        cap = int(self.cfg.get("sample_vocab_cap", 0) or 0)
        if cap:
            cap = min(cap, len(self.llm.get_tokenizer()))
        self._allowed_ids = list(range(cap)) if cap else None

    def ready(self):
        return True

    def vocab_size(self):
        return self.llm.llm_engine.get_model_config().get_vocab_size()

    def generate_batch(self, prompt_ids_list, max_tokens_list):
        """One continuation per prompt; returns list[list[int]] token ids."""
        from vllm import SamplingParams, TokensPrompt

        prompts, params = [], []
        for ids, mt in zip(prompt_ids_list, max_tokens_list):
            prompts.append(TokensPrompt(prompt_token_ids=list(ids)))
            params.append(
                SamplingParams(
                    max_tokens=int(mt),
                    temperature=float(self.cfg.get("temperature", 1.0)),
                    top_p=float(self.cfg.get("top_p", 1.0)),
                    repetition_penalty=1.0,
                    allowed_token_ids=self._allowed_ids,
                )
            )
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in outs]
