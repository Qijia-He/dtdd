"""Write the WebShop task files (train / test splits) used by the experiments.

A WebShop "task" is just a session id: the environment resets to
``session=<task_id>`` and presents the corresponding human goal. The 120k
catalog built by ``build_webshop_catalog.py`` yields 12,251 goals, numbered
``0 .. 12250``. The experiments hold out 2,000 of them for evaluation and
train on the remaining 10,251.

The held-out ids were drawn once and are shipped verbatim in
``../data/webshop_test_ids.json`` (a sorted JSON list of 2,000 integers); the
training split is their complement. Shipping the list rather than a seed
keeps the split exact regardless of how any random generator changes.

Output
------
Two JSON-Lines files in ``--output-dir``, one task per line, in the schema
the TCOD task reader expects:

    {"task_id": <int>, "target": ""}

    train.jsonl     10,251 lines
    test.jsonl       2,000 lines

Point a YAML config's taskset ``path`` at these files.
"""

import argparse
import json
import os
from typing import List

# Goals in the 120k catalog are numbered 0 .. NUM_GOALS - 1.
NUM_GOALS = 12_251
EXPECTED_TEST_SIZE = 2_000

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEST_IDS = os.path.join(_HERE, "..", "data", "webshop_test_ids.json")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write the splits to."
    )
    parser.add_argument(
        "--test-ids",
        default=DEFAULT_TEST_IDS,
        help="JSON list of held-out task ids.",
    )
    parser.add_argument(
        "--num-goals",
        type=int,
        default=NUM_GOALS,
        help="Total number of goals; ids run from 0 to num_goals - 1.",
    )
    return parser.parse_args()


def write_split(path: str, task_ids: List[int]) -> None:
    """Write one task per line in the reader's schema."""
    with open(path, "w") as handle:
        for task_id in task_ids:
            handle.write(json.dumps({"task_id": task_id, "target": ""}) + "\n")


def main() -> None:
    args = parse_args()
    with open(args.test_ids) as handle:
        test_ids: List[int] = sorted(json.load(handle))

    assert len(test_ids) == len(set(test_ids)), "duplicate test ids"
    assert all(0 <= task_id < args.num_goals for task_id in test_ids), (
        "a test id lies outside 0 .. num_goals - 1"
    )
    test_set = set(test_ids)
    train_ids = [i for i in range(args.num_goals) if i not in test_set]

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.jsonl")
    test_path = os.path.join(args.output_dir, "test.jsonl")
    write_split(train_path, train_ids)
    write_split(test_path, test_ids)

    print(f"train: {len(train_ids)} -> {train_path}")
    print(f"test:  {len(test_ids)} -> {test_path}")
    if len(test_ids) != EXPECTED_TEST_SIZE:
        print(
            f"note: {len(test_ids)} test ids, not the {EXPECTED_TEST_SIZE} "
            "used in the paper"
        )


if __name__ == "__main__":
    main()
