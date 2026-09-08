"""Smoke tests for the code lane: imports, registry, config, and data shape.

These run without a GPU and without a sandbox. They check the things that
break silently -- a workflow that is not registered, a config whose DTDD
block was renamed, a data file whose schema drifted -- not that training
converges.

Run from the repository root, after ``./setup.sh --code``::

    CODE_ENV/bin/python -m pytest recipe_dtdd_code/tests -q
"""

import json
import os
import pathlib

import pytest

pytest.importorskip("verl", reason="run ./setup.sh --code first")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RECIPE_ROOT = REPO_ROOT / "third_party" / "SOD" / "recipe" / "dtdd"
DATA_DIR = pathlib.Path(
    os.environ.get("CODE_DATA_DIR", REPO_ROOT / "datasets" / "code")
)


def test_recipe_is_installed_into_the_sod_checkout():
    """setup.sh copies the recipe into the base checkout; check it landed."""
    assert RECIPE_ROOT.is_dir(), (
        f"{RECIPE_ROOT} missing -- run ./setup.sh --code"
    )
    for name in ("main_dtdd.py", "dtdd_agent_loop.py", "dtdd_ray_trainer.py",
                 "dtdd_dp_actor.py", "dtdd_fsdp_workers.py",
                 "teacher_vllm_server.py", "run_dtdd_code.sh"):
        assert (RECIPE_ROOT / name).is_file(), f"{name} missing"


def test_dtdd_modules_import():
    """The DTDD classes import, which also exercises the overlay."""
    from recipe.dtdd.dtdd_agent_loop import DTDDToolAgentLoop
    from recipe.dtdd.dtdd_dp_actor import DTDDDataParallelPPOActor
    from recipe.dtdd.dtdd_fsdp_workers import DTDDAsyncActorRolloutRefWorker
    from recipe.dtdd.dtdd_ray_trainer import DTDDRayPPOTrainer

    # Each must extend its stock counterpart, or the pinned-copy methods
    # would silently stop overriding anything.
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

    assert issubclass(DTDDToolAgentLoop, ToolAgentLoop)
    assert issubclass(DTDDRayPPOTrainer, RayPPOTrainer)
    assert issubclass(DTDDDataParallelPPOActor, DataParallelPPOActor)
    assert issubclass(
        DTDDAsyncActorRolloutRefWorker, AsyncActorRolloutRefWorker
    )


def test_agent_loop_is_registered_over_tool_agent():
    """The loop must replace the stock "tool_agent" entry, not add a name."""
    import yaml

    entries = yaml.safe_load(
        (RECIPE_ROOT / "config" / "agent_loop.yaml").read_text()
    )
    assert len(entries) == 1
    assert entries[0]["name"] == "tool_agent"
    assert entries[0]["_target_"] == (
        "recipe.dtdd.dtdd_agent_loop.DTDDToolAgentLoop"
    )


def test_trainer_config_has_the_paper_defaults():
    """The dtdd block's keys and the values the paper's arms rely on."""
    import yaml

    config = yaml.safe_load(
        (RECIPE_ROOT / "config" / "dtdd_trainer.yaml").read_text()
    )
    dtdd = config["dtdd"]
    assert dtdd["takeover_mode"] == "rewind"
    # Pure distillation on the prefix; the GRPO term is an ablation only.
    assert dtdd["grpo_prefix_coef"] == 0.0
    assert dtdd["suffix_sft_coef"] == 1.0
    assert dtdd["suffix_loss_type"] == "ce"
    # Level 2 sleep would discard the frozen teacher weights for good.
    assert dtdd["teacher"]["sleep_level"] == 1
    # vLLM V1 rejects prompt_logprobs together with prefix caching, and
    # scoring is what the divergence trigger is built on.
    assert dtdd["teacher"]["enable_prefix_caching"] is False
    rollout = config["actor_rollout_ref"]["rollout"]
    assert rollout["calculate_log_probs"] is True


def test_the_two_arms_differ_only_in_the_trigger():
    """The launcher's DTDD and OPD branches must differ in tau/epsilon only."""
    script = (RECIPE_ROOT / "run_dtdd_code.sh").read_text()
    # The baseline runs the same code path with a threshold that never fires.
    assert "DTDD_TAU" in script and "DTDD_EPSILON_RANDOM" in script
    # w_k weighting off by default: the code lane's OPD baseline is vanilla
    # per-token OPD, the same definition the other lanes use.
    assert "stepwise_enable=${STEPWISE_ENABLE:-False}" in script


@pytest.mark.skipif(
    not (DATA_DIR / "code_slice_train.parquet").exists(),
    reason="build the code datasets first (see the README)",
)
def test_data_slices_have_the_expected_shape():
    """Row counts, sources and the held-out split's disjointness."""
    import pyarrow.parquet as pq

    train = pq.read_table(DATA_DIR / "code_slice_train.parquet")
    heldout = pq.read_table(DATA_DIR / "code_slice_heldout200.parquet")
    heldout_full = pq.read_table(
        DATA_DIR / "code_slice_heldout200_fulltests.parquet"
    )
    minus = pq.read_table(DATA_DIR / "code_slice_train_minus200.parquet")

    assert train.num_rows == 4205
    assert heldout.num_rows == heldout_full.num_rows == 200
    assert minus.num_rows == train.num_rows - heldout.num_rows

    sources = set(train.column("data_source").to_pylist())
    assert all(s.startswith("train-code-") for s in sources)
    assert set(heldout.column("data_source").to_pylist()) == {
        "heldout200_code_sub"
    }
    assert set(heldout_full.column("data_source").to_pylist()) == {
        "heldout200_code_full"
    }

    # minus200 must not contain any held-out problem.
    held_indices = {
        row["index"] for row in heldout.column("extra_info").to_pylist()
    }
    minus_indices = {
        row["index"] for row in minus.column("extra_info").to_pylist()
    }
    assert held_indices.isdisjoint(minus_indices)

    # The full-test copy of a problem must grade on at least as many cases as
    # the subsampled copy; that difference is what the two sets measure.
    def n_tests(row) -> int:
        ground_truth = json.loads(row["reward_model"]["ground_truth"])
        if "inputs" in ground_truth:
            return len(ground_truth["inputs"])
        return len(
            [x for x in ground_truth["test_code"].split("\n") if x]
        ) - 1

    sub_rows = heldout.to_pylist()
    full_rows = heldout_full.to_pylist()
    assert all(
        n_tests(f) >= n_tests(s) for s, f in zip(sub_rows, full_rows)
    )


@pytest.mark.skipif(
    not (DATA_DIR / "lcb_v6_fixed.parquet").exists(),
    reason="build the LiveCodeBench parquets first (see the README)",
)
def test_lcb_prompt_and_grading_tests_are_separate():
    """The defect the rebuilt files exist to fix must stay fixed."""
    import pyarrow.parquet as pq

    rows = pq.read_table(DATA_DIR / "lcb_v6_fixed.parquet").to_pylist()
    assert rows, "empty parquet"
    graded_only_on_shown = 0
    for row in rows:
        shown = json.loads(row["public_tests"])
        graded = json.loads(row["reward_model"])["ground_truth"]
        assert graded["inputs"], "a problem with no grading tests"
        if graded["inputs"] == shown["inputs"]:
            graded_only_on_shown += 1
    # Some problems legitimately have only public cases; most must not.
    assert graded_only_on_shown < len(rows) / 2, (
        f"{graded_only_on_shown}/{len(rows)} problems are graded exactly on "
        "the cases printed into their prompt -- the split did not happen"
    )
