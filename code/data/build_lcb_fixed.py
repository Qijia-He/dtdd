"""Build a LiveCodeBench evaluation parquet with prompt and grading tests split.

The LiveCodeBench file that ships with ``Gen-Verse/Open-AgentRL-Eval``
(``lcb_v6_2502_2505.parquet``) carries exactly one test case per problem, and
``recipe/demystify/reward.py`` uses that same case twice: printed into the
prompt as the worked example and handed to ``compute_score`` as the grading
truth. A program that ignores the question and prints the shown output
scores 1.0. Showing public examples is normal and stays; grading only on the
shown example is the defect.

Upstream LiveCodeBench keeps the two apart: ``public_test_cases`` (plain
JSON) and ``private_test_cases`` (base64 of zlib of pickle of JSON). This
script writes both:

``public_tests``
    JSON string, for the prompt only.
``reward_model``
    ``{"ground_truth": {"inputs": [...], "outputs": [...]}}`` over private
    cases first, then public, for grading only. The reward code reads
    ``public_tests`` when the column is present and falls back to the old
    behaviour otherwise, so the historical file still loads.

Only stdin/stdout problems are kept. LiveCodeBench's functional (LeetCode)
problems need the ``import_prefix`` / ``entry_point`` shape that the other
branch of ``compute_score`` expects, and mixing the two silently would be
worse than excluding them.

The graded set is chosen in two stages, the two passes the paper's files
went through. Stage one takes the first ``STAGE_ONE_CAP`` (50) cases
public-first and applies ``--max-bytes-per-problem``; stage two reorders
the survivors private-first, takes ``--n-tests-cap`` of them and applies the
byte cap again. The count cap is the one that matters for cost -- every
grading case is one sandbox execution, and ten blind cases already settle
whether a problem was solved. The byte cap keeps the file loadable: a few
problems ship megabyte-scale stdin blobs, and the evaluator holds the whole
parquet in memory. Private cases come first so the graded set is maximally
unseen; public cases only top up a short tail.

The paper's files: ``test.jsonl`` -> v1 (219 problems, 2023-05..2024-03),
``test2.jsonl`` -> v2 (57), ``test3.jsonl`` -> v3 (53), ``test4.jsonl`` -> v4
(65), ``test6.jsonl`` -> v6 (112, 2025-01..2025-04), all with the defaults
below and no problem falling back to public-only tests.

Usage::

    python build_lcb_fixed.py --release test6.jsonl \
        --data-source LiveCodeBench_v6_fixed --output lcb_v6_fixed.parquet

``--release`` is downloaded from ``livecodebench/code_generation_lite`` on
the Hugging Face Hub unless ``--raw-dir`` already holds it. Some releases
are over a gigabyte and a few problems carry 57 MB of compressed tests that
peak in the gigabytes when unpickled; run this on a machine with real
memory, one release at a time.
"""

import argparse
import base64
import collections
import json
import os
import pickle
import statistics
import zlib

import pyarrow as pa
import pyarrow.parquet as pq

HF_DATASET = "livecodebench/code_generation_lite"
# Stage-one case cap, public cases first (see the module docstring).
STAGE_ONE_CAP = 50


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--release",
        required=True,
        help="Raw release file, e.g. test.jsonl (v1) or test6.jsonl (v6).",
    )
    parser.add_argument(
        "--data-source",
        required=True,
        help="Value of the data_source column, e.g. LiveCodeBench_v6_fixed. "
        "Must start with 'LiveCodeBench' for the reward code to route it.",
    )
    parser.add_argument("--output", required=True, help="Output parquet.")
    parser.add_argument(
        "--raw-dir",
        default="lcb_raw",
        help="Where raw releases are downloaded to / read from.",
    )
    parser.add_argument(
        "--max-private-bytes",
        type=int,
        default=200_000_000,
        help="Skip a problem's private tests when the encoded blob is larger "
        "than this, keeping only its public tests and marking the row "
        "private_skipped (default 200 MB: nothing skipped).",
    )
    parser.add_argument(
        "--n-tests-cap",
        type=int,
        default=10,
        help="Grading cases per problem, private first (default 10).",
    )
    parser.add_argument(
        "--max-bytes-per-problem",
        type=int,
        default=200_000,
        help="Stop adding cases once their cumulative input+output size "
        "passes this, keeping at least one (default 200 kB).",
    )
    return parser.parse_args()


def decode_private(blob: str, max_bytes: int) -> tuple[list, bool]:
    """Decode a private_test_cases field.

    Args:
        blob: Plain JSON, or base64(zlib(pickle(JSON))).
        max_bytes: Skip decoding when the encoded blob is larger.

    Returns:
        The list of cases and whether decoding was skipped or failed.
    """
    if not blob:
        return [], False
    if len(blob) > max_bytes:
        return [], True
    try:
        return json.loads(blob), False
    except Exception:
        try:
            decoded = pickle.loads(zlib.decompress(base64.b64decode(blob)))
            return json.loads(decoded), False
        except Exception:
            return [], True


def cap_bytes(cases: list, byte_cap: int) -> list:
    """Keep a prefix of ``cases`` whose input+output size fits ``byte_cap``.

    At least one case is always kept, so a single oversized case survives.

    Args:
        cases: Cases as ``{"input", "output", ...}`` dicts, in priority order.
        byte_cap: Maximum cumulative size in bytes.

    Returns:
        The kept prefix.
    """
    kept, total = [], 0
    for case in cases:
        size = len(case.get("input", "")) + len(case.get("output", ""))
        if kept and total + size > byte_cap:
            break
        kept.append(case)
        total += size
    return kept


def select_graded(survivors: list, n_public: int, n_cap: int,
                  byte_cap: int) -> tuple[list, int, bool]:
    """Stage two: choose the graded cases from the stage-one survivors.

    The paper's files went through two passes, kept separate here so the
    output is identical to them. Stage one (in ``main``) takes the first
    ``STAGE_ONE_CAP`` cases public-first and applies the byte cap, which can
    push a large private case out behind a small public one. This stage
    reorders the survivors private-first, takes ``n_cap`` of them and applies
    the byte cap again.

    Args:
        survivors: The stage-one cases, public ones first.
        n_public: How many leading survivors are public.
        n_cap: Maximum number of graded cases.
        byte_cap: Maximum cumulative input+output size.

    Returns:
        The graded cases, how many of them are private (unseen), and whether
        any survivor was dropped.
    """
    private_first = survivors[n_public:]
    public_tail = survivors[:n_public]
    candidates = private_first[:n_cap]
    candidates += public_tail[: max(0, n_cap - len(candidates))]
    graded = cap_bytes(candidates, byte_cap)
    n_unseen = min(len(private_first), len(graded))
    return graded, n_unseen, len(graded) < len(survivors)


def main() -> None:
    args = parse_args()
    raw_path = os.path.join(args.raw_dir, args.release)
    if not os.path.exists(raw_path):
        from huggingface_hub import hf_hub_download

        raw_path = hf_hub_download(
            HF_DATASET, args.release, repo_type="dataset",
            local_dir=args.raw_dir,
        )
    print(f"reading {raw_path} ({os.path.getsize(raw_path) / 1e6:.0f} MB)")

    rows = []
    skipped = collections.Counter()
    with open(raw_path) as f:
        for line in f:
            problem = json.loads(line)
            public = (
                json.loads(problem["public_test_cases"])
                if problem.get("public_test_cases")
                else []
            )
            private, private_skipped = decode_private(
                problem.get("private_test_cases"), args.max_private_bytes
            )
            # Stage one: public first, capped by count and then by bytes.
            survivors = cap_bytes(
                (public + private)[:STAGE_ONE_CAP], args.max_bytes_per_problem
            )
            if not survivors:
                skipped["no_tests"] += 1
                continue
            if any(c.get("testtype") != "stdin" for c in survivors):
                skipped["functional"] += 1
                continue
            n_public = min(len(public), len(survivors))
            graded, n_unseen, trimmed = select_graded(
                survivors, n_public, args.n_tests_cap,
                args.max_bytes_per_problem,
            )
            rows.append({
                "data_source": args.data_source,
                "problem": (
                    problem.get("question_title", "") + "\n"
                    + problem["question_content"]
                ).strip(),
                # Prompt only; never scored against.
                "public_tests": json.dumps({
                    "inputs": [c["input"] for c in public],
                    "outputs": [c["output"] for c in public],
                }),
                # Grading only.
                "reward_model": json.dumps({"ground_truth": {
                    "inputs": [c["input"] for c in graded],
                    "outputs": [c["output"] for c in graded],
                }}),
                "ability": "code",
                "extra_info": json.dumps({
                    "question_id": problem.get("question_id"),
                    "difficulty": problem.get("difficulty"),
                    "platform": problem.get("platform"),
                    "contest_date": str(problem.get("contest_date")),
                    "n_public": len(public),
                    "n_total": len(graded),
                    "n_unseen": n_unseen,
                    "trimmed": trimmed,
                    "private_skipped": private_skipped,
                    "index": len(rows),
                }),
            })
            del private

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output)
    infos = [json.loads(r["extra_info"]) for r in rows]
    print(f"wrote {args.output}: {len(rows)} problems, skipped {dict(skipped)}")
    print(
        f"  grading cases/problem: median "
        f"{statistics.median(i['n_total'] for i in infos):.0f}, "
        f"unseen median {statistics.median(i['n_unseen'] for i in infos):.0f}"
    )
    print(
        f"  private tests skipped on "
        f"{sum(i['private_skipped'] for i in infos)} problems; difficulty "
        f"{dict(collections.Counter(i['difficulty'] for i in infos))}"
    )


if __name__ == "__main__":
    main()
