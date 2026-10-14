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
"""DTDD worker: swaps ``DataParallelPPOActor`` for ``DTDDDataParallelPPOActor``.

The re-wrap reuses the FSDP module and optimizer already built by
``super().init_model()`` (``fsdp_workers.py`` at YoungZ365/SOD@110c4b8
constructs ``DataParallelPPOActor(config=..., actor_module=
self.actor_module_fsdp, actor_optimizer=self.actor_optimizer)``), so nothing
about model construction changes. Only ``update_policy`` differs, and only
when the DTDD masks are present in the batch.
"""

from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

from .dtdd_dp_actor import DTDDDataParallelPPOActor


class DTDDAsyncActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """``AsyncActorRolloutRefWorker`` whose actor applies the DTDD loss."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self) -> None:
        super().init_model()
        if self._is_actor and getattr(self, "actor", None) is not None:
            self.actor = DTDDDataParallelPPOActor(
                config=self.actor.config,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )
            print("DTDD worker: actor re-wrapped with DTDDDataParallelPPOActor")
