"""Label every DAPO-Math-17k problem with the student's pass rate, then cut hard subsets.

The training experiments need a "hard" slice of DAPO-Math-17k: the problems a given
student model rarely or never solves. Difficulty here is measured against a specific
model rather than assumed, so the slice has to be recomputed whenever the student
changes.

Pipeline
--------
1. Build a vLLM engine for the student model named by ``DIFFICULTY_MODEL``.
2. Read the source problems, optionally subsampling for a quick trial run.
3. Sample ``SAMPLES_PER_PROBLEM`` completions per problem at temperature 1.0.
4. Grade each completion against the problem's ground truth; the pass rate is the
   fraction of that problem's completions judged correct.
5. Write every problem with its pass rate.
6. Write one subset per difficulty threshold, keeping problems at or below it.

Input
-----
``DIFFICULTY_SOURCE`` -- a Parquet file, one row per problem, with at least:

    prompt         chat-style turns, e.g. ``[{"role": "user", "content": "..."}]``;
                   a bare string is also accepted and used as-is
    reward_model   a mapping containing ``ground_truth``, the reference answer

All other columns are carried through untouched.

Output
------
``DIFFICULTY_OUTPUT``          every input row plus a float ``pass_rate`` column in
                               [0, 1], the fraction of sampled completions that were
                               correct
``<output_dir>/dapo_hard0``    rows with ``pass_rate <= 0.0`` -- never solved
``<output_dir>/dapo_hard25``   rows with ``pass_rate <= 0.25``

The two subsets drop ``pass_rate`` again so they are drop-in replacements for the
source file in a training config.

Import order
------------
vLLM must be imported and its engine constructed before anything that initialises a
CUDA context in this process. Importing verl pulls in torch, which does exactly that,
and the engine's multiprocessing workers then fail to start. Everything else is
imported at module level as usual; the single verl import is deferred into
``_load_scorer`` for this reason and this reason only.
"""

import os
import sys

# Must be set before vLLM starts its workers.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams

# --- model and engine ---
STUDENT_MODEL_PATH = os.environ["DIFFICULTY_MODEL"]
TENSOR_PARALLEL_SIZE = int(os.environ.get("DIFFICULTY_TENSOR_PARALLEL", "4"))
GPU_MEMORY_UTILIZATION = 0.85
MAX_MODEL_LENGTH = 8192

# --- data ---
SOURCE_PARQUET = os.environ.get("DIFFICULTY_SOURCE", "datasets/dapo-math-17k.parquet")
OUTPUT_PARQUET = os.environ.get("DIFFICULTY_OUTPUT", "datasets/dapo_difficulty.parquet")
SUBSAMPLE_SIZE = int(os.environ.get("DIFFICULTY_SUBSAMPLE", "0"))  # 0 = use every problem

# --- sampling ---
SAMPLES_PER_PROBLEM = int(os.environ.get("DIFFICULTY_SAMPLES", "4"))
MAX_NEW_TOKENS = int(os.environ.get("DIFFICULTY_MAX_TOKENS", "4096"))
SAMPLING_TEMPERATURE = 1.0
RANDOM_SEED = 0

# --- subsets: (upper bound on pass rate, file name stem) ---
DIFFICULTY_THRESHOLDS = [(0.0, "dapo_hard0"), (0.25, "dapo_hard25")]

# Extra packages (e.g. the grader's dependencies) that are not on the default path.
EXTRA_PACKAGES_PATH = os.environ.get("DIFFICULTY_EXTRA_PACKAGES", "")
VERL_REPO_PATH = os.environ.get("DIFFICULTY_VERL_PATH", "verl")


def _load_scorer():
    """Import verl's grader.

    Deferred rather than imported at module level: verl imports torch, torch
    initialises a CUDA context, and a CUDA context in the parent process before the
    vLLM engine exists stops the engine's workers from starting. Call this only
    after ``build_engine``.
    """
    if EXTRA_PACKAGES_PATH:
        sys.path.insert(0, EXTRA_PACKAGES_PATH)
    sys.path.insert(0, VERL_REPO_PATH)
    from verl.utils.reward_score.ttrl_math import compute_score

    return compute_score


def build_engine():
    """Construct the vLLM engine. Must run before any torch/verl import."""
    return LLM(
        model=STUDENT_MODEL_PATH,
        dtype="bfloat16",
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LENGTH,
        enforce_eager=True,
        trust_remote_code=True,
        seed=RANDOM_SEED,
    )


def load_problems():
    """Read the source Parquet, optionally subsampling for a trial run."""
    problems = pd.read_parquet(SOURCE_PARQUET)
    if SUBSAMPLE_SIZE and SUBSAMPLE_SIZE < len(problems):
        problems = problems.sample(n=SUBSAMPLE_SIZE, random_state=RANDOM_SEED)
        problems = problems.reset_index(drop=True)
    return problems


def extract_prompt_text(prompt_field):
    """Return the user text from a chat-style prompt, or the field itself if it is a string."""
    if isinstance(prompt_field, (list, tuple, np.ndarray)):
        return prompt_field[0]["content"]
    return prompt_field


def compute_pass_rates(engine, prompts, ground_truths):
    """Sample completions per prompt and return the fraction graded correct for each."""
    compute_score = _load_scorer()
    sampling_params = SamplingParams(
        n=SAMPLES_PER_PROBLEM,
        temperature=SAMPLING_TEMPERATURE,
        top_p=1.0,
        max_tokens=MAX_NEW_TOKENS,
    )
    generations = engine.generate(prompts, sampling_params, use_tqdm=True)

    pass_rates = []
    for generation, ground_truth in zip(generations, ground_truths):
        correct = sum(
            1
            for completion in generation.outputs
            if compute_score(completion.text, str(ground_truth))["acc"]
        )
        pass_rates.append(correct / SAMPLES_PER_PROBLEM)
    return pass_rates


def write_hard_subsets(problems, output_dir):
    """Write one Parquet per threshold, keeping problems at or below that pass rate."""
    for threshold, name in DIFFICULTY_THRESHOLDS:
        subset = problems[problems["pass_rate"] <= threshold]
        path = os.path.join(output_dir, f"{name}.parquet")
        subset.drop(columns=["pass_rate"]).to_parquet(path)
        print(f"  {name}: {len(subset)} problems -> {path}", flush=True)


def main():
    engine = build_engine()

    problems = load_problems()
    print(
        f"tagging {len(problems)} problems, {SAMPLES_PER_PROBLEM} samples each",
        flush=True,
    )
    prompts = [extract_prompt_text(p) for p in problems["prompt"]]
    ground_truths = [entry["ground_truth"] for entry in problems["reward_model"]]

    problems["pass_rate"] = compute_pass_rates(engine, prompts, ground_truths)
    problems.to_parquet(OUTPUT_PARQUET)

    pass_rates = np.array(problems["pass_rate"])
    print(
        f"pass_rate mean={pass_rates.mean():.3f} "
        f"never_solved={(pass_rates == 0).mean():.3f} "
        f"at_most_25pct={(pass_rates <= 0.25).mean():.3f}",
        flush=True,
    )

    write_hard_subsets(problems, os.path.dirname(OUTPUT_PARQUET) or ".")
    print("TAG_DONE", flush=True)


if __name__ == "__main__":
    main()
