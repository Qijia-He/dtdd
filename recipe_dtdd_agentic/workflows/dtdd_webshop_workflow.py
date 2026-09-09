"""Divergence-Triggered Dynamic Distillation (DTDD) workflow for WebShop.

The DTDD turn loop -- student samples, teacher scores, and the teacher takes
the turn over when the per-turn divergence exceeds ``tau`` -- lives in
``dtdd_base_workflow.py``, which carries the full rationale, the token
alignment convention and the tensor shapes. This module supplies only what is
specific to WebShop: the prompt templates, the action menu and its validation,
the environment's scalar reward, and the re-scoring guard below.

Unlike ALFWorld, a WebShop environment is expensive to build and is therefore
created once per workflow and reset by session id; ``run_async`` rolls out
``repeat_times`` episodes of the same session (``can_repeat = True``).

What differs from the ALFWorld sibling
--------------------------------------
Re-scoring a teacher turn feeds the prompt plus the teacher's whole turn back
through BOTH models. WebShop observations are long enough that this can
exceed ``max_model_len``, and in an earlier version the resulting error
escaped ``run_async`` and discarded the entire episode. Because only
triggered episodes could raise, the surviving batch was silently filtered to
the low-divergence ones. ``_rescore_teacher_turn`` below therefore (a) skips a
takeover whose prompt would exceed ``max_rescore_tokens`` before paying for
it, and (b) catches any re-scoring failure and falls back to the student's
turn -- which is exactly a non-triggered turn. Both cases are counted and
reported as the ``takeover_skipped_len`` and ``takeover_skipped_err``
metrics: if either is non-zero, the arm is not doing what its name says.

That guard was added after the ALFWorld runs and is deliberately not moved
into the base class, so the ALFWorld workflow keeps reproducing its published
numbers as they were produced.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from torch import Tensor

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, Task
from trinity.common.workflows.envs.TCOD.dtdd_base_workflow import (
    DTDDAgenticWorkflow,
)
from trinity.common.workflows.envs.TCOD.webshop.utils import (
    HISTORY_LENGTH,
    WEBSHOP_TEMPLATE,
    WEBSHOP_TEMPLATE_NO_HIS,
    _create_webshop_env,
    _extract_task_description,
    _format_available_actions,
    _format_history,
    format_observation,
    parse_action,
    validate_action,
)

# Registry key. Referenced by name from YAML (``default_workflow_type``) and by
# the tests, so it is deliberately kept identical to the original release.
WORKFLOW_NAME = "DTDD_webshop_workflow"

# Defaults for ``task.workflow_args``; the rest are in the base class.
DEFAULT_MAX_ENV_STEPS = 15
DEFAULT_MAX_RESCORE_TOKENS = 10240

# A WebShop episode succeeds when the environment's final reward is 1.0; the
# comparison tolerates float noise.
SUCCESS_REWARD_THRESHOLD = 0.999


@WORKFLOWS.register_module(WORKFLOW_NAME)
class DTDDWebshopWorkflow(DTDDAgenticWorkflow):
    """DTDD workflow for WebShop.

    Use ``advantage_fn: multi_turn_opd``, the same as OPD. See
    ``DTDDAgenticWorkflow`` for the shared ``task.workflow_args``, plus:

        max_env_steps (int): maximum episode length in environment steps.
            Default 15.
        max_rescore_tokens (int): a takeover whose prompt plus teacher turn
            is longer than this, in tokens, is skipped before re-scoring.
            Default 10240.
    """

    can_repeat: bool = True

    WORKFLOW_NAME: str = WORKFLOW_NAME
    DEFAULT_MAX_ENV_STEPS: int = DEFAULT_MAX_ENV_STEPS

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ) -> None:
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )
        self.max_rescore_tokens: int = int(
            task.workflow_args.get(
                "max_rescore_tokens", DEFAULT_MAX_RESCORE_TOKENS
            )
        )
        self._takeover_skipped_len: int = 0
        self._takeover_skipped_err: int = 0
        self._session_id: int = 0
        self._available_actions: Dict[str, Any] = {}
        # One environment per workflow: building it is expensive, and episodes
        # differ only by the session it is reset to.
        self.env = _create_webshop_env()

    def reset(self, task: Task) -> None:
        """Reset the workflow with a new task, including its rollout count."""
        super().reset(task)
        self.repeat_times = task.repeat_times or 1

    async def run_async(self) -> List[Experience]:
        """Run ``repeat_times`` episodes of one session and concatenate them."""
        self._session_id = int(self.task_desc)
        all_turn_responses: List[Experience] = []

        for rollout_index in range(self.repeat_times):
            rollout_responses = await self._run_episode(
                self.env, run_id=self.run_id_base + rollout_index
            )
            all_turn_responses.extend(rollout_responses)

        return all_turn_responses

    # --------------------------------------------------------------- hooks --
    def _extract_task_description(self, observation: str) -> str:
        return _extract_task_description(observation)

    def _format_observation(self, observation: str) -> str:
        return format_observation(observation)

    def _reset_env(self, env: Any) -> str:
        """Reset the persistent environment to this episode's session."""
        env.reset(session=self._session_id)
        self._env_done = False
        self._env_rounds = 0
        self._final_reward = 0.0
        return env.observation

    def _format_actions(self, env: Any) -> str:
        """Render the action menu, keeping the raw one for validation."""
        self._available_actions = env.get_available_actions()
        return _format_available_actions(self._available_actions)

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
            return WEBSHOP_TEMPLATE_NO_HIS.format(
                task_description=task_description,
                current_observation=formatted_observation,
                available_actions=formatted_actions,
            )
        action_history = "\n".join(history[-HISTORY_LENGTH:])
        return WEBSHOP_TEMPLATE.format(
            task_description=task_description,
            step_count=turn_index,
            history_length=min(HISTORY_LENGTH, len(history)),
            action_history=action_history,
            current_step=turn_index + 1,
            current_observation=formatted_observation,
            available_actions=formatted_actions,
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
        """Validate the action, then step the environment with it.

        An invalid action does not advance the environment: the error message
        becomes the next observation, so the model can correct itself.
        """
        action = parse_action(response_text)
        action_valid, error_message = validate_action(
            action, self._available_actions
        )
        history.append(
            _format_history(formatted_observation, turn_index + 1, action)
        )
        if action_valid:
            observation, reward, done, _ = env.step(action)
            return observation, reward, done
        return error_message, 0.0, False

    def _finalize_episode(
        self, *, turn_index: Any, done: bool, reward: float
    ) -> None:
        if done:
            self._env_done = True
            self._env_rounds = turn_index + 1
            self._final_reward = float(reward)
        else:
            # Exhausted ``max_env_steps`` without finishing; the reward stays
            # at the 0.0 set when the episode began.
            self._env_rounds = self.max_env_steps

    def _episode_metrics(self) -> Dict[str, float]:
        """WebShop's scalar score, its success indicator, and the skip counts.

        The two skip counters are first-class metrics on purpose: an
        exception that only ever appears in a stack trace is an exception
        nobody sees.
        """
        return {
            "session_id": float(self._session_id),
            "score": float(self._final_reward),
            "success": (
                1.0
                if self._final_reward >= SUCCESS_REWARD_THRESHOLD
                else 0.0
            ),
            "takeover_skipped_len": float(self._takeover_skipped_len),
            "takeover_skipped_err": float(self._takeover_skipped_err),
        }

    async def _rescore_teacher_turn(
        self, teacher_response: Experience, teacher_response_length: int
    ) -> Tuple[Optional[Tensor], Optional[Tensor], int]:
        """Re-score a teacher turn, skipping the ones that would break.

        See the module docstring for why this guard exists here and not in
        the base class.

        Args:
            teacher_response: The teacher's generated turn.
            teacher_response_length: Its response length in tokens.

        Returns:
            The student's log-probs on the teacher's tokens, the teacher's
            log-probs on its own tokens, and the response length -- or
            ``(None, None, 0)`` when the turn falls back to the student's.
        """
        # Skip a takeover that could not be re-scored anyway, before paying
        # for the two extra log-prob passes.
        if len(teacher_response.tokens) > self.max_rescore_tokens:
            self._takeover_skipped_len += 1
            return None, None, 0

        # Never let a re-scoring failure discard the episode. The turn falls
        # back to the student's own, which is exactly a non-triggered turn:
        # it costs one takeover, keeps the sample, and keeps the batch
        # unbiased.
        try:
            return await super()._rescore_teacher_turn(
                teacher_response, teacher_response_length
            )
        except Exception as error:  # noqa: BLE001
            self._takeover_skipped_err += 1
            self.logger.warning(
                "[DTDD] takeover re-score failed, falling back to the "
                "student turn (%s: %s)",
                type(error).__name__,
                str(error)[:160],
            )
            return None, None, 0
