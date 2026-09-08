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
"""DTDD entry point: the SOD ``main_ppo`` run flow wired to the DTDD classes.

This is a pinned copy of ``verl/trainer/main_ppo.py`` (YoungZ365/SOD@110c4b8)
with three deltas:

* the hydra config is ``recipe/dtdd/config/dtdd_trainer.yaml``, which extends
  the stock ``ppo_trainer`` config with a top-level ``dtdd`` block;
* ``Role.ActorRollout`` is served by ``DTDDAsyncActorRolloutRefWorker``, whose
  actor adds the suffix-SFT term to the policy loss;
* the trainer is ``DTDDRayPPOTrainer``, which owns the frozen teacher pool and
  turns the per-token origin mask into DTDD loss masks and metrics.

Launch from the SOD checkout root so that ``recipe.*`` is importable::

    python3 -m recipe.dtdd.main_dtdd <hydra overrides>

``recipe/dtdd/run_dtdd_code.sh`` is the override bundle used for the paper.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import is_cuda_available


@hydra.main(config_path="config", config_name="dtdd_trainer", version_base=None)
def main(config) -> None:
    run_ppo(config)


def run_ppo(config) -> None:
    """Initialise Ray and run the training task on a dedicated actor.

    Args:
        config: The resolved hydra config (``dtdd_trainer.yaml`` + overrides).
    """
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create(
            {**ray_init_kwargs, "runtime_env": runtime_env}
        )
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), (
            "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        )
        nsys_config = config.global_profiler.global_tool_config.nsys
        nsight_options = OmegaConf.to_container(
            nsys_config.controller_nsight_options
        )
        runner = TaskRunner.options(
            runtime_env={"nsight": nsight_options}
        ).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """Builds the worker mapping and runs ``DTDDRayPPOTrainer``.

    Imports of worker classes stay inside the methods, as in the upstream
    ``main_ppo.TaskRunner``: the class is pickled to a Ray actor, and importing
    the GPU worker modules at module load would pull torch and vLLM into the
    driver process before Ray has set up the runtime environment.
    """

    def __init__(self) -> None:
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Register the DTDD actor-rollout worker (FSDP + async rollout only).

        Args:
            config: The resolved training config.

        Returns:
            The worker class and the Ray worker-group class to drive it.
        """
        from verl.single_controller.ray import RayWorkerGroup

        assert config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}, (
            "DTDD recipe supports the FSDP strategy only"
        )
        assert config.actor_rollout_ref.rollout.mode == "async", (
            "DTDD recipe requires rollout.mode=async"
        )

        from recipe.dtdd.dtdd_fsdp_workers import DTDDAsyncActorRolloutRefWorker

        actor_rollout_cls = DTDDAsyncActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(
            actor_rollout_cls
        )

        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config) -> None:
        """Register the critic worker matching the configured strategy."""
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            use_legacy_worker_impl = config.trainer.get(
                "use_legacy_worker_impl", "auto"
            )
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker

                print("Using new worker implementation")
            else:
                raise ValueError(
                    f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}"
                )

        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)

    def init_resource_pool_mgr(self, config):
        """Create the single global resource pool every role is mapped onto."""
        from verl.trainer.ppo.ray_trainer import Role

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node]
            * config.trainer.nnodes,
        }
        self.mapping[Role.ActorRollout] = global_pool_id
        self.mapping[Role.Critic] = global_pool_id
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=self.mapping
        )
        return resource_pool_manager

    def add_reward_model_worker(self, config) -> None:
        """Register a reward-model worker when one is enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            self.role_worker_mapping[Role.RewardModel] = ray.remote(
                RewardModelWorker
            )
            self.mapping[Role.RewardModel] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls) -> None:
        """Register the reference-policy worker.

        DTDD delta: ``dtdd.enable`` also requires the reference worker, because
        the update-phase ``ref_log_prob`` comes from the teacher weights loaded
        into it (see ``DTDDRayPPOTrainer.init_workers`` for the path check).

        Args:
            config: The resolved training config.
            ref_policy_cls: The worker class to use for the reference policy.
        """
        from verl.trainer.ppo.ray_trainer import Role

        token_kl_reg = getattr(config.algorithm, "token_kl_reg", None)
        token_kl_needed = bool(
            token_kl_reg and getattr(token_kl_reg, "enable", False)
        )
        dtdd_needed = bool(config.get("dtdd", None) and config.dtdd.enable)
        if (
            config.algorithm.use_kl_in_reward
            or config.actor_rollout_ref.actor.use_kl_loss
            or token_kl_needed
            or dtdd_needed
        ):
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(
                ref_policy_cls
            )
            self.mapping[Role.RefPolicy] = "global_pool"

    def run(self, config) -> None:
        """Assemble datasets, reward functions and workers, then train."""
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(
            f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}"
        )
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(
            config
        )
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(
                self.role_worker_mapping
            ),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(
            local_path, trust_remote_code=trust_remote_code
        )
        processor = hf_processor(
            local_path, trust_remote_code=trust_remote_code, use_fast=True
        )

        reward_fn = load_reward_manager(
            config,
            tokenizer,
            num_examine=0,
            **config.reward_model.get("reward_kwargs", {}),
        )
        val_reward_fn = load_reward_manager(
            config,
            tokenizer,
            num_examine=1,
            **config.reward_model.get("reward_kwargs", {}),
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files, config.data, tokenizer, processor,
            is_train=True,
        )
        val_dataset = create_rl_dataset(
            config.data.val_files, config.data, tokenizer, processor,
            is_train=False,
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        from recipe.dtdd.dtdd_ray_trainer import DTDDRayPPOTrainer

        trainer = DTDDRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
