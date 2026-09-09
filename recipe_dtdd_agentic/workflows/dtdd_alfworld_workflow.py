"""Divergence-Triggered Dynamic Distillation (DTDD) workflow for ALFWorld.

The DTDD turn loop -- student samples, teacher scores, and the teacher takes
the turn over when the per-turn divergence exceeds ``tau`` -- lives in
``dtdd_base_workflow.py``, which carries the full rationale, the token
alignment convention and the tensor shapes. This module supplies only what is
specific to ALFWorld: the prompt templates, the admissible-command menu, and
the episode's success reward.

An ALFWorld episode owns its environment: ``run_async`` builds one from the
task's game file and closes it afterwards, and rolls out a single episode per
task (``can_repeat = False``).

A note on the WebShop sibling
-----------------------------
``dtdd_webshop_workflow.py`` overrides ``_rescore_teacher_turn`` to add a
guard: if re-scoring a teacher turn fails (over-long prompt, transient engine
error) it falls back to the student's turn and counts the skip, instead of
letting the exception drop the whole episode. That guard was added after the
WebShop runs and is deliberately NOT inherited here -- this workflow keeps the
base class's unguarded re-scoring, so it reproduces the ALFWorld numbers in
the paper as they were produced.
"""

from __future__ import annotations

from typing import Any, List, Tuple

from trinity.common.experience import Experience
from trinity.common.workflows import WORKFLOWS
from trinity.common.workflows.envs.TCOD.alfworld.utils import (
    ALFWORLD_TEMPLATE,
    ALFWORLD_TEMPLATE_NO_HIS,
    HISTORY_LENGTH,
    _create_alfworld_env,
    _extract_task,
    _format_history,
    format_observation,
    parse_action,
)
from trinity.common.workflows.envs.TCOD.dtdd_base_workflow import (
    DTDDAgenticWorkflow,
)

# Registry key. Referenced by name from YAML (``default_workflow_type``) and by
# the tests, so it is deliberately kept identical to the original release.
WORKFLOW_NAME = "DTDD_alfworld_workflow"

# Default for ``task.workflow_args``; the rest are in the base class.
DEFAULT_MAX_ENV_STEPS = 30

# ALFWorld reports no scalar reward, only whether the task was completed.
SUCCESS_REWARD = 1.0


@WORKFLOWS.register_module(WORKFLOW_NAME)
class DTDDAlfworldWorkflow(DTDDAgenticWorkflow):
    """DTDD workflow for ALFWorld.

    Use ``advantage_fn: multi_turn_opd`` (``MultiTurnOpdAdvantage``), the same
    as OPD. See ``DTDDAgenticWorkflow`` for the recognized
    ``task.workflow_args``; ``max_env_steps`` defaults to 30 here.
    """

    can_repeat: bool = False

    WORKFLOW_NAME: str = WORKFLOW_NAME
    DEFAULT_MAX_ENV_STEPS: int = DEFAULT_MAX_ENV_STEPS

    async def run_async(self) -> List[Experience]:
        """Run one episode and return one experience per turn."""
        game_file_path = self.task_desc
        env = _create_alfworld_env(game_file_path)
        try:
            return await self._run_episode(env, run_id=self.run_id_base)
        finally:
            env.close()

    # --------------------------------------------------------------- hooks --
    def _extract_task_description(self, observation: str) -> str:
        return _extract_task(observation)

    def _format_observation(self, observation: str) -> str:
        return format_observation(observation)

    def _reset_env(self, env: Any) -> str:
        """Reset the environment and keep the first ``info`` for the menu."""
        observation, self._info = env.reset()
        self._env_done = False
        self._env_rounds = 0
        return observation

    def _format_actions(self, env: Any) -> str:
        """Render this turn's admissible commands.

        ALFWorld reports them in ``info``, sometimes wrapped in a
        single-element batch dimension. ``help`` is dropped: it is always
        admissible and never useful as an action.
        """
        del env  # The menu comes from the last step's info, not the env.
        admissible_commands = self._info.get("admissible_commands", [])
        if admissible_commands and isinstance(admissible_commands[0], list):
            admissible_commands = admissible_commands[0]
        return "\n ".join(
            f"'{command}'"
            for command in admissible_commands
            if command != "help"
        )

    def _build_user_content(
        self,
        *,
        task_description: str,
        turn_index: int,
        history: List[str],
        formatted_observation: str,
        formatted_actions: str,
    ) -> str:
        if len(history) < HISTORY_LENGTH:
            return ALFWORLD_TEMPLATE_NO_HIS.format(
                current_observation=formatted_observation,
                admissible_actions=formatted_actions,
            )
        action_history = "\n".join(history[-HISTORY_LENGTH:])
        return ALFWORLD_TEMPLATE.format(
            task_description=task_description,
            step_count=turn_index,
            history_length=min(HISTORY_LENGTH, len(history)),
            action_history=action_history,
            current_step=turn_index + 1,
            current_observation=formatted_observation,
            admissible_actions=formatted_actions,
        )

    def _execute_action(
        self,
        env: Any,
        *,
        response_text: str,
        formatted_observation: str,
        turn_index: int,
        history: List[str],
    ) -> Tuple[str, float, bool]:
        """Step the environment with the parsed action.

        ALFWorld's per-step reward is ignored; the episode reward is a
        completion indicator set in ``_finalize_episode``.
        """
        action = parse_action(response_text)
        history.append(
            _format_history(formatted_observation, turn_index + 1, action)
        )
        observation, _reward, done, self._info = env.step(action)
        return observation, 0.0, done

    def _finalize_episode(
        self, *, turn_index: Any, done: bool, reward: float
    ) -> None:
        del reward  # ALFWorld scores completion, not the environment's reward.
        if done:
            self._env_done = True
            self._env_rounds = turn_index + 1
            self._final_reward = SUCCESS_REWARD
        else:
            # Exhausted ``max_env_steps`` without finishing: failure.
            self._env_rounds = self.max_env_steps
            self._final_reward = 0.0
