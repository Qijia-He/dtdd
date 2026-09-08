"""Divergence-Triggered Dynamic Distillation (DTDD) workflow for WebShop.

This forks the plain on-policy distillation (OPD) WebShop workflow that ships
with TCOD and mirrors ``dtdd_alfworld_workflow.py``, which carries the full
rationale. In summary:

Each turn the student samples its action and the teacher scores the student's
action tokens, giving the per-turn divergence

    turn_divergence = mean_t |logp_student(t) - logp_teacher(t)|.

If ``turn_divergence > tau`` (and the turn index is at least
``min_prefix_turns``, and this is not an evaluation episode), the TEACHER
regenerates that turn. Its action is executed in the environment and the
recorded experience is filled so the student trains to imitate the teacher:

    recorded.logprobs         = student logp on the TEACHER's tokens
                                (this becomes ``old_log_probs``)
    recorded.teacher_logprobs = teacher logp on the TEACHER's tokens

A non-triggered (student) turn is identical to plain OPD:

    recorded.logprobs         = student sampling logp on the student's tokens
    recorded.teacher_logprobs = teacher logp on the student's tokens

Both turn types emit the same experience schema, so the existing
``multi_turn_opd`` advantage, ``kl_coef * (teacher_logprobs - old_log_probs)``,
needs no changes. Evaluation is student-only (the trigger is gated on
``not self.is_eval``), so the OPD-vs-DTDD comparison stays fair.

Token alignment: student and teacher share the Qwen2.5 tokenizer; whichever
model generated a turn defines the response boundary
(``response_start = prompt_length - 1``), mirroring OPD's
teacher-scores-student slicing exactly.

Tensor shapes
-------------
All per-turn log-probability tensors are 1-D over the turn's response tokens:

    R = number of response tokens in the turn

    student_logprobs:  [R]   logp of each response token under the student
    teacher_logprobs:  [R]   logp of each response token under the teacher

What differs from the ALFWorld sibling
--------------------------------------
Re-scoring a teacher turn feeds the prompt plus the teacher's whole turn back
through BOTH models. WebShop observations are long enough that this can
exceed ``max_model_len``, and in an earlier version the resulting error
escaped ``run_async`` and discarded the entire episode. Because only
triggered episodes could raise, the surviving batch was silently filtered to
the low-divergence ones. This workflow therefore (a) skips a takeover whose
prompt would exceed ``max_rescore_tokens`` before paying for it, and (b)
catches any re-scoring failure and falls back to the student's turn -- which
is exactly a non-triggered turn. Both cases are counted and reported as the
``takeover_skipped_len`` and ``takeover_skipped_err`` metrics: if either is
non-zero, the arm is not doing what its name says. The ALFWorld workflow does
not carry this guard, so it reproduces the ALFWorld numbers as produced.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from torch import Tensor

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, Task, Workflow
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

if TYPE_CHECKING:
    # gym / web_agent_site are optional heavy dependencies that the
    # environment factory imports lazily; only the type checker needs the name.
    from gym import Env as GymEnv

# Registry key. Referenced by name from YAML (``default_workflow_type``) and by
# the tests, so it is deliberately kept identical to the original release.
WORKFLOW_NAME = "DTDD_webshop_workflow"

# Defaults for ``task.workflow_args``.
DEFAULT_TAU = 0.7
DEFAULT_MIN_PREFIX_TURNS = 1
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_ENV_STEPS = 15
DEFAULT_MAX_RESCORE_TOKENS = 10240

# A WebShop episode succeeds when the environment's final reward is 1.0; the
# comparison tolerates float noise.
SUCCESS_REWARD_THRESHOLD = 0.999


@WORKFLOWS.register_module(WORKFLOW_NAME)
class DTDDWebshopWorkflow(Workflow):
    """DTDD workflow for WebShop.

    Use ``advantage_fn: multi_turn_opd``, the same as OPD.

    Recognized ``task.workflow_args``:

        tau (float): divergence threshold; the teacher takes over the turn
            when ``turn_divergence > tau``. Default 0.7.
        min_prefix_turns (int): earliest 1-based turn index eligible for a
            takeover. Default 1 (every turn is eligible).
        temperature (float): temperature for the scoring (log-prob) calls.
            Default 1.0.
        max_env_steps (int): maximum episode length in environment steps.
            Default 15.
        max_rescore_tokens (int): a takeover whose prompt plus teacher turn
            is longer than this, in tokens, is skipped before re-scoring.
            Default 10240.
    """

    is_async: bool = True
    can_reset: bool = True
    can_repeat: bool = True

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
        self.reset(task)

        assert (
            self.auxiliary_model_wrappers is not None
            and len(self.auxiliary_model_wrappers) >= 1
        ), "DTDD requires at least one auxiliary model to act as the teacher."
        self.teacher_model: ModelWrapper = self.auxiliary_model_wrappers[0]

        workflow_args = task.workflow_args
        self.temperature: float = float(
            workflow_args.get("temperature", DEFAULT_TEMPERATURE)
        )
        self.max_env_steps: int = int(
            workflow_args.get("max_env_steps", DEFAULT_MAX_ENV_STEPS)
        )
        self.tau: float = float(workflow_args.get("tau", DEFAULT_TAU))
        self.min_prefix_turns: int = int(
            workflow_args.get("min_prefix_turns", DEFAULT_MIN_PREFIX_TURNS)
        )
        self.max_rescore_tokens: int = int(
            workflow_args.get("max_rescore_tokens", DEFAULT_MAX_RESCORE_TOKENS)
        )
        self._takeover_skipped_len: int = 0
        self._takeover_skipped_err: int = 0
        self.env: GymEnv = _create_webshop_env()

        # Episode state, populated by ``_run_episode``.
        self._env_done: bool = False
        self._env_rounds: int = 0
        self._final_reward: float = 0.0
        self.run_id_base: int = 0

    def reset(self, task: Task) -> None:
        """Reset the workflow with a new task."""
        self.task = task
        self.format_args = task.format_args
        self.raw_task = task.raw_task
        self.task_desc: str = task.task_desc or "0"
        self.is_eval: bool = task.is_eval
        self.repeat_times: int = task.repeat_times or 1

    def set_repeat_times(self, repeat_times: int, run_id_base: int) -> None:
        """Record the rollout count and the base run id used for their eids."""
        self.repeat_times = repeat_times
        self.task.rollout_args.n = repeat_times
        self.run_id_base = run_id_base

    def compute_reward(self, response: Experience) -> float:
        """Return the episode-level reward, shared by all of its turns."""
        del response  # The reward is per-episode, not per-turn.
        return self._final_reward

    @property
    def rollout_args(self) -> Dict[str, Any]:
        """The task's generation config as a plain dict of kwargs."""
        return asdict(self.task.rollout_args)

    def format_messages(self) -> List[Dict[str, str]]:
        """Return the initial chat messages; WebShop episodes start empty."""
        return []

    @staticmethod
    def _compute_turn_divergence(
        student_logprobs: Tensor,
        teacher_logprobs: Tensor,
    ) -> float:
        """Compute the per-turn divergence between student and teacher.

        This is ``mean_t |logp_student(t) - logp_teacher(t)|`` over the turn's
        response tokens.

        Args:
            student_logprobs: [R] log-probs of the response tokens under the
                student.
            teacher_logprobs: [R] log-probs of the same tokens under the
                teacher.

        Returns:
            The mean absolute log-prob gap, or 0.0 for an empty turn.
        """
        if len(student_logprobs) == 0:
            return 0.0
        return float((student_logprobs - teacher_logprobs).abs().mean().item())

    async def run_async(self) -> List[Experience]:
        """Run ``repeat_times`` episodes of one session and concatenate them."""
        session_id = int(self.task_desc)
        all_turn_responses: List[Experience] = []

        for rollout_index in range(self.repeat_times):
            rollout_responses = await self._run_episode(
                session_id=session_id,
                run_id=self.run_id_base + rollout_index,
            )
            all_turn_responses.extend(rollout_responses)

        return all_turn_responses

    async def _run_episode(  # noqa: C901
        self,
        session_id: int,
        run_id: int,
    ) -> List[Experience]:
        """Roll out one WebShop episode with divergence-triggered takeover.

        Args:
            session_id: The WebShop session (goal) to reset the environment to.
            run_id: The run id stamped on every experience's eid.

        Returns:
            One ``Experience`` per executed turn, in order. Every experience
            carries ``logprobs`` and ``teacher_logprobs`` in the schema that
            the ``multi_turn_opd`` advantage expects; the last one also
            carries the episode-level metrics.
        """
        self.env.reset(session=session_id)
        observation = self.env.observation
        self._env_done = False
        self._env_rounds = 0
        self._final_reward = 0.0

        task_description = _extract_task_description(observation)
        history: List[str] = []
        memory = self.format_messages()
        turn_responses: List[Experience] = []

        generation_kwargs: Dict[str, Any] = {**self.rollout_args, "n": 1}
        if generation_kwargs.get("logprobs") is None:
            generation_kwargs["logprobs"] = 0

        per_turn_kl_sums: List[float] = []
        turn_divergences: List[float] = []
        takeover_count = 0

        for turn_index in range(self.max_env_steps):
            available_actions = self.env.get_available_actions()
            formatted_observation = format_observation(observation)
            formatted_available_actions = _format_available_actions(
                available_actions
            )

            if len(history) < HISTORY_LENGTH:
                user_content = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=task_description,
                    current_observation=formatted_observation,
                    available_actions=formatted_available_actions,
                )
            else:
                action_history = "\n".join(history[-HISTORY_LENGTH:])
                user_content = WEBSHOP_TEMPLATE.format(
                    task_description=task_description,
                    step_count=turn_index,
                    history_length=min(HISTORY_LENGTH, len(history)),
                    action_history=action_history,
                    current_step=turn_index + 1,
                    current_observation=formatted_observation,
                    available_actions=formatted_available_actions,
                )

            # The assistant message is appended only after the student-vs-
            # teacher decision below, so the chat history matches the action
            # that was actually executed in the environment.
            memory = memory + [{"role": "user", "content": user_content}]

            # Step 1: the student samples this turn, exactly as in OPD.
            student_responses = await self.model.chat_async(
                memory, **generation_kwargs
            )
            student_response = student_responses[0]
            if student_response.logprobs is None:
                raise RuntimeError(
                    f"{WORKFLOW_NAME} requires the student model to return "
                    "logprobs. Set rollout_args.logprobs (e.g. 0) in the task "
                    "config."
                )

            # Step 2: the teacher scores the student's turn. This is both the
            # OPD signal for a student turn and the DTDD trigger statistic.
            teacher_logprobs_full = await self.teacher_model.logprobs_async(
                tokens=student_response.tokens.tolist(),
                temperature=self.temperature,
            )
            student_response_start = student_response.prompt_length - 1
            teacher_logprobs = teacher_logprobs_full[student_response_start:]
            student_logprobs = student_response.logprobs
            assert len(teacher_logprobs) == len(student_logprobs), (
                f"[DTDD] student-turn length mismatch: "
                f"teacher={len(teacher_logprobs)}, "
                f"student={len(student_logprobs)}, "
                f"tokens={len(student_response.tokens)}, "
                f"prompt_length={student_response.prompt_length}"
            )

            turn_divergence = self._compute_turn_divergence(
                student_logprobs, teacher_logprobs
            )
            turn_divergences.append(turn_divergence)

            # Evaluation episodes are always student-only, so that OPD and
            # DTDD are compared on the same footing.
            should_take_over = (
                not self.is_eval
                and turn_divergence > self.tau
                and (turn_index + 1) >= self.min_prefix_turns
                and len(student_logprobs) > 0
            )

            recorded_response = student_response
            is_teacher_turn = False

            # Step 3 (on trigger): the teacher generates the turn instead.
            if should_take_over:
                teacher_responses = await self.teacher_model.chat_async(
                    memory, **generation_kwargs
                )
                teacher_response = teacher_responses[0]
                teacher_response_length = (
                    len(teacher_response.tokens)
                    - teacher_response.prompt_length
                )

                # Skip a takeover that could not be re-scored anyway, before
                # paying for the two extra log-prob passes.
                if (
                    teacher_response_length > 0
                    and len(teacher_response.tokens) > self.max_rescore_tokens
                ):
                    self._takeover_skipped_len += 1
                    teacher_response_length = 0

                if teacher_response_length > 0:
                    teacher_response_start = teacher_response.prompt_length - 1

                    # Never let a re-scoring failure discard the episode. The
                    # turn falls back to the student's own, which is exactly
                    # a non-triggered turn: it costs one takeover, keeps the
                    # sample, and keeps the batch unbiased.
                    try:
                        # Student log-probs on the teacher's tokens: this
                        # becomes ``old_log_probs`` for the imitation update.
                        student_on_teacher_full = (
                            await self.model.logprobs_async(
                                tokens=teacher_response.tokens.tolist(),
                                temperature=self.temperature,
                            )
                        )
                        # Teacher log-probs on its own tokens: the imitation
                        # target.
                        teacher_on_teacher_full = (
                            await self.teacher_model.logprobs_async(
                                tokens=teacher_response.tokens.tolist(),
                                temperature=self.temperature,
                            )
                        )
                    except Exception as error:  # noqa: BLE001
                        self._takeover_skipped_err += 1
                        self.logger.warning(
                            "[DTDD] takeover re-score failed, falling back to "
                            "the student turn (%s: %s)",
                            type(error).__name__,
                            str(error)[:160],
                        )
                        teacher_response_length = 0
                        student_on_teacher_full = None
                        teacher_on_teacher_full = None

                if teacher_response_length > 0:
                    student_on_teacher = student_on_teacher_full[
                        teacher_response_start:
                    ]
                    teacher_on_teacher = teacher_on_teacher_full[
                        teacher_response_start:
                    ]

                    assert (
                        len(student_on_teacher)
                        == len(teacher_on_teacher)
                        == teacher_response_length
                    ), (
                        f"[DTDD] teacher-turn length mismatch: "
                        f"student_on_teacher={len(student_on_teacher)}, "
                        f"teacher_on_teacher={len(teacher_on_teacher)}, "
                        f"response_length={teacher_response_length}, "
                        f"tokens={len(teacher_response.tokens)}, "
                        f"prompt_length={teacher_response.prompt_length}"
                    )

                    teacher_response.logprobs = student_on_teacher
                    teacher_response.teacher_logprobs = teacher_on_teacher
                    recorded_response = teacher_response
                    is_teacher_turn = True
                    takeover_count += 1
                # An empty or skipped teacher turn falls back to the student.

            if not is_teacher_turn:
                # A non-triggered turn is identical to OPD.
                student_response.teacher_logprobs = teacher_logprobs

            # Append the assistant message for the action that will execute.
            response_text = recorded_response.response_text or ""
            memory.append({"role": "assistant", "content": response_text})

            if recorded_response.logprobs is None:
                raise RuntimeError(
                    f"{WORKFLOW_NAME} requires logprobs on the recorded turn."
                )
            turn_responses.append(recorded_response)

            kl_sum = (
                recorded_response.logprobs - recorded_response.teacher_logprobs
            ).sum().item()
            per_turn_kl_sums.append(kl_sum)

            if recorded_response.metrics is None:
                recorded_response.metrics = {}
            # Metric keys are the on-disk telemetry schema read by the
            # harvest and plotting tools; they are not renamed.
            recorded_response.metrics["if_teacher"] = float(is_teacher_turn)
            recorded_response.metrics["d_k"] = turn_divergence

            # Execute the recorded action; WebShop validates it first.
            action = parse_action(response_text)
            action_valid, error_message = validate_action(
                action, available_actions
            )
            history.append(
                _format_history(formatted_observation, turn_index + 1, action)
            )

            if action_valid:
                observation, reward, done, _ = self.env.step(action)
            else:
                observation = error_message
                reward = 0.0
                done = False

            if done:
                self._env_done = True
                self._env_rounds = turn_index + 1
                self._final_reward = float(reward)
                break
        else:
            # Exhausted ``max_env_steps`` without finishing; the reward stays
            # at the 0.0 set when the episode began.
            self._env_rounds = self.max_env_steps

        # Finalize: episode reward and eids.
        for step_index, response in enumerate(turn_responses):
            if response.metrics is None:
                response.metrics = {}
            response.reward = self.compute_reward(response)
            response.eid.run = run_id
            response.eid.step = step_index

        trajectory_kl_divergence = sum(per_turn_kl_sums)
        num_turns = len(turn_responses)
        takeover_rate = (
            (takeover_count / num_turns) if num_turns > 0 else 0.0
        )

        if turn_responses:
            last_response = turn_responses[-1]
            if last_response.metrics is None:
                last_response.metrics = {}
            last_response.metrics["env_rounds"] = self._env_rounds
            last_response.metrics["env_done"] = 1.0 if self._env_done else 0.0
            last_response.metrics["kl_divergence"] = trajectory_kl_divergence
            last_response.metrics["session_id"] = float(session_id)
            last_response.metrics["score"] = float(self._final_reward)
            last_response.metrics["success"] = (
                1.0 if self._final_reward >= SUCCESS_REWARD_THRESHOLD else 0.0
            )
            # DTDD telemetry. The two skip counters are first-class metrics
            # on purpose: an exception that only ever appears in a stack
            # trace is an exception nobody sees.
            last_response.metrics["takeover_rate"] = takeover_rate
            last_response.metrics["takeover_skipped_len"] = float(
                self._takeover_skipped_len
            )
            last_response.metrics["takeover_skipped_err"] = float(
                self._takeover_skipped_err
            )
            last_response.metrics["teacher_env_rounds"] = takeover_count
            last_response.metrics["student_env_rounds"] = (
                num_turns - takeover_count
            )
            last_response.metrics["mean_d_k"] = (
                sum(turn_divergences) / len(turn_divergences)
                if turn_divergences
                else 0.0
            )

        return turn_responses
