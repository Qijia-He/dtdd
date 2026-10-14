"""Smoke tests for the agentic DTDD lane: import, registry, config validation.

These run without a GPU and without the environments installed. They check
the three things that break silently when the overlay is misapplied:

1. The workflow module imports and defines the expected class.
2. The TCOD workflow registry resolves the name the YAML configs use.
3. A shipped config loads and passes ``check_and_update``, with the DTDD
   knobs (workflow type, ``tau``) surviving validation.

``check_and_update`` normally calls ``create_dummy_lora``, which instantiates
the full student model to build a placeholder adapter. That is a side effect
needing GPU memory, not config-validation logic, so it is stubbed out here.

Run from the repository root after ``./setup.sh``::

    DTDD_ROOT=$(pwd) python -m pytest alfworld/tests -q

The tests skip, rather than fail, when ``trinity`` is not importable, so the
suite is safe to run in an environment that has only the math lane set up.
"""

import importlib
import os
from typing import Any

import pytest

trinity = pytest.importorskip("trinity")  # noqa: F841

DTDD_ROOT = os.environ.get(
    "DTDD_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)
CONFIG_DIR = os.path.join(DTDD_ROOT, "alfworld", "configs")

# (env, registry name, module path, class name, config file)
LANES = [
    (
        "alfworld",
        "DTDD_alfworld_workflow",
        "trinity.common.workflows.envs.TCOD.alfworld.DTDD_workflow",
        "DTDDAlfworldWorkflow",
        os.path.join(CONFIG_DIR, "dtdd_lora.yaml"),
    ),
]


BASE_MODULE = "trinity.common.workflows.envs.TCOD.dtdd_base_workflow"
BASE_CLASS = "DTDDAgenticWorkflow"


def test_base_workflow_imports() -> None:
    """The shared DTDD turn loop is installed next to the env packages."""
    module = importlib.import_module(BASE_MODULE)
    assert hasattr(module, BASE_CLASS), f"{BASE_MODULE} lacks {BASE_CLASS}"


@pytest.mark.parametrize(
    "env, registry_name, module_path, class_name, config_path", LANES
)
def test_workflow_subclasses_base(
    env: str,
    registry_name: str,
    module_path: str,
    class_name: str,
    config_path: str,
) -> None:
    """Both environment workflows share the base class's turn loop."""
    del env, registry_name, config_path
    base = getattr(importlib.import_module(BASE_MODULE), BASE_CLASS)
    workflow = getattr(importlib.import_module(module_path), class_name)
    assert issubclass(workflow, base), (
        f"{class_name} does not extend {BASE_CLASS}"
    )


@pytest.mark.parametrize(
    "env, registry_name, module_path, class_name, config_path", LANES
)
def test_workflow_imports(
    env: str,
    registry_name: str,
    module_path: str,
    class_name: str,
    config_path: str,
) -> None:
    """The overlay installs the workflow module and it defines the class."""
    del env, registry_name, config_path
    module = importlib.import_module(module_path)
    assert hasattr(module, class_name), f"{module_path} lacks {class_name}"


@pytest.mark.parametrize(
    "env, registry_name, module_path, class_name, config_path", LANES
)
def test_registry_resolves(
    env: str,
    registry_name: str,
    module_path: str,
    class_name: str,
    config_path: str,
) -> None:
    """The name the YAML uses resolves through TCOD's workflow registry."""
    del env, config_path
    from trinity.common.workflows import WORKFLOWS

    resolved = WORKFLOWS.get(registry_name)
    assert resolved is not None, f"registry returned None for {registry_name}"
    assert resolved.__module__ == module_path
    assert resolved.__name__ == class_name


@pytest.mark.parametrize(
    "env, registry_name, module_path, class_name, config_path", LANES
)
def test_config_validates(
    env: str,
    registry_name: str,
    module_path: str,
    class_name: str,
    config_path: str,
    tmp_path: Any,
    monkeypatch: Any,
) -> None:
    """A shipped config loads and survives check_and_update with DTDD intact."""
    del env, module_path, class_name
    if not os.path.exists(config_path):
        pytest.skip(f"config not rendered: {config_path}")

    import trinity.common.config_validator as config_validator
    from trinity.common.config import load_config

    def fake_create_dummy_lora(
        model_path: str,
        checkpoint_job_dir: str,
        lora_rank: int,
        lora_alpha: int,
        target_modules: Any,
        exclude_modules: Any = None,
        **kwargs: Any,
    ) -> str:
        del model_path, lora_rank, lora_alpha, target_modules
        del exclude_modules, kwargs
        dummy_dir = os.path.join(checkpoint_job_dir, "dummy_lora")
        os.makedirs(dummy_dir, exist_ok=True)
        return dummy_dir

    monkeypatch.setattr(
        config_validator, "create_dummy_lora", fake_create_dummy_lora
    )

    config = load_config(config_path)
    config.checkpoint_root_dir = str(tmp_path)  # never write to the real root
    validated = config.check_and_update()  # raising here means invalid

    taskset = validated.buffer.explorer_input.tasksets[0]
    assert taskset.default_workflow_type == registry_name
    assert "tau" in taskset.workflow_args, "tau missing from workflow_args"
    assert 0.0 < float(taskset.workflow_args["tau"]) < 1.0
    assert validated.algorithm.advantage_fn == "multi_turn_opd"
    assert validated.algorithm.algorithm_type == "on_policy_distill"
