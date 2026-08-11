"""DTDD entry: main_ppo with the trainer class swapped INSIDE the remote
TaskRunner process.

TaskRunner runs as a Ray remote actor in its own worker process, which imports
verl.trainer.main_ppo fresh — a driver-side module patch does NOT propagate.
So we subclass TaskRunner and apply the patch at the top of run(), i.e. inside
the remote process, before super().run() resolves the module-global
`RayPPOTrainer` symbol. recipe_dtdd/ rides in on PYTHONPATH (inherited by the
local raylet's workers; single-node assumption).

With DTDD_ENABLE=0 this entry is bit-identical to `python -m verl.trainer.main_ppo`.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import hydra
import ray

import verl.trainer.main_ppo as m

CONFIG_DIR = os.path.abspath(os.path.join(_HERE, "..", "verl", "verl", "trainer", "config"))


class DTDDTaskRunner(m.TaskRunner):
    def run(self, config):
        import sys as _sys

        if _HERE not in _sys.path:
            _sys.path.insert(0, _HERE)
        import verl.trainer.main_ppo as mm
        from dtdd_trainer import DTDDRayPPOTrainer

        mm.RayPPOTrainer = DTDDRayPPOTrainer
        print("DTDD entry: RayPPOTrainer swapped -> DTDDRayPPOTrainer (in TaskRunner process)")
        return super().run(config)


@hydra.main(config_path=CONFIG_DIR, config_name="ppo_trainer", version_base=None)
def main(config):
    m.run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(DTDDTaskRunner))


if __name__ == "__main__":
    main()
