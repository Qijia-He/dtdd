"""Build the code-lane training and held-out parquets from Open-AgentRL-30K.

The code lane trains on the code-only slice of ``Gen-Verse/Open-AgentRL-30K``
(4,205 ``train-code-*`` rows) with each problem's grading tests subsampled to
at most ``--max-tests`` cases, and evaluates on 200 problems held out from
that slice. This script writes all four files the launchers read:

``code_slice_train.parquet``
    Every ``train-code-*`` row, tests subsampled with ``random.Random(0)``.
``code_slice_heldout200.parquet``
    The 200 held-out problems with the same subsampled tests
    (``data_source`` = ``heldout200_code_sub``).
``code_slice_heldout200_fulltests.parquet``
    The same 200 problems with their complete original test suites
    (``data_source`` = ``heldout200_code_full``; about 45 tests per problem
    versus 7 after subsampling).
``code_slice_train_minus200.parquet``
    The training slice with the 200 held-out problems removed.

Why subsample tests in the data rather than in the reward code: code-only
training makes every response hit the execution sandbox at reward time, and
the 30K mix hid that cost (89% of its rows are math or science with a
string-match reward). Capping tests here keeps the reward code, which is
shared with evaluation and must stay full-fidelity, untouched.

The held-out split is shipped as the 200 row indices into the source parquet
(``heldout200_indices.json``) rather than as a seed, so it is exact. The
``test-code-leetcode-*`` rows of the source are a held-out split in the
source dataset itself and are excluded on principle.

Usage::

    python build_code_slices.py --source Open-AgentRL-30K.parquet \
        --indices heldout200_indices.json --output-dir datasets/code
"""

import argparse
import collections
import json
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq

TRAIN_PREFIX = "train-code-"
HELDOUT_SUB_SOURCE = "heldout200_code_sub"
HELDOUT_FULL_SOURCE = "heldout200_code_full"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--source", required=True, help="Open-AgentRL-30K.parquet."
    )
    parser.add_argument(
        "--indices",
        required=True,
        help="JSON list of the 200 held-out row indices into --source.",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Where the four parquets go."
    )
    parser.add_argument(
        "--max-tests",
        type=int,
        default=12,
        help="Keep at most this many grading tests per problem (default 12).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Subsampling seed (default 0)."
    )
    return parser.parse_args()


def subsample_tests(ground_truth: dict, max_tests: int,
                    rng: random.Random) -> dict:
    """Cap a problem's grading tests, in place, to ``max_tests`` cases.

    Two ground-truth shapes occur in the source. TACO-style rows carry
    parallel ``inputs`` / ``outputs`` lists; LeetCode-style rows carry a
    ``test_code`` string whose first line is the ``def check(candidate):``
    header and whose later lines are one assertion each.

    Args:
        ground_truth: The parsed ``reward_model.ground_truth`` dict.
        max_tests: The cap.
        rng: The shared random source; consumed only when a row is capped,
            in source order, which is what makes the result reproducible.

    Returns:
        The same dict, possibly with fewer tests.
    """
    if "inputs" in ground_truth:
        n = len(ground_truth["inputs"])
        if n > max_tests:
            keep = sorted(rng.sample(range(n), max_tests))
            ground_truth["inputs"] = [ground_truth["inputs"][i] for i in keep]
            ground_truth["outputs"] = [ground_truth["outputs"][i] for i in keep]
    elif "test_code" in ground_truth:
        lines = [x for x in ground_truth["test_code"].split("\n") if x != ""]
        header, asserts = lines[0], lines[1:]
        if len(asserts) > max_tests:
            keep = sorted(rng.sample(range(len(asserts)), max_tests))
            asserts = [asserts[i] for i in keep]
            ground_truth["test_code"] = "\n".join([header] + asserts)
    return ground_truth


def main() -> None:
    args = parse_args()
    with open(args.indices) as f:
        heldout_indices = set(json.load(f))
    rows = pq.read_table(args.source).to_pylist()
    rng = random.Random(args.seed)

    train, heldout_sub, heldout_full = [], [], []
    for row in rows:
        if not str(row["data_source"]).startswith(TRAIN_PREFIX):
            continue
        # The source rows are numbered by position in extra_info.index.
        index = int(row["extra_info"]["index"])
        full_ground_truth = row["reward_model"]["ground_truth"]
        ground_truth = subsample_tests(
            json.loads(full_ground_truth), args.max_tests, rng
        )
        row["reward_model"]["ground_truth"] = json.dumps(ground_truth)
        train.append(row)
        if index in heldout_indices:
            sub = dict(row)
            sub["data_source"] = HELDOUT_SUB_SOURCE
            sub["extra_info"] = {
                "index": index, "orig_source": row["data_source"]
            }
            heldout_sub.append(sub)
            full = dict(sub)
            full["data_source"] = HELDOUT_FULL_SOURCE
            full["reward_model"] = dict(row["reward_model"])
            full["reward_model"]["ground_truth"] = full_ground_truth
            heldout_full.append(full)

    assert len(heldout_sub) == len(heldout_indices), (
        f"found {len(heldout_sub)} held-out rows for {len(heldout_indices)} "
        "indices: wrong --source file?"
    )
    train_minus = [
        r for r in train if int(r["extra_info"]["index"]) not in heldout_indices
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    outputs = {
        "code_slice_train.parquet": train,
        "code_slice_heldout200.parquet": heldout_sub,
        "code_slice_heldout200_fulltests.parquet": heldout_full,
        "code_slice_train_minus200.parquet": train_minus,
    }
    for name, table_rows in outputs.items():
        path = os.path.join(args.output_dir, name)
        pq.write_table(pa.Table.from_pylist(table_rows), path)
        print(f"wrote {len(table_rows):>5} rows -> {path}")
    counts = collections.Counter(r["data_source"] for r in train)
    print("train sources:", counts.most_common())


if __name__ == "__main__":
    main()
