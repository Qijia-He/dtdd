#!/usr/bin/env python3
"""Auto-submit a held-out eval as soon as a training checkpoint is complete.

Motivation: every coverage hole in our experiment record came from "train now,
evaluate later" -- one arm's clean checkpoints were never graded at all,
another lost 4 grid cells to a disk cleanup before anyone evaluated them.
Evaluating at save time makes a checkpoint's *result* durable even if the
checkpoint itself is later pruned (the results store is kilobytes; the
checkpoints are terabytes).

Optional SLURM automation -- nothing in the training or eval pipeline depends
on it. Run it from cron / a monitor loop / by hand. It is idempotent: a ledger
records every (arm, step) it has submitted, so re-running only ever picks up
genuinely new checkpoints.

  tools/autoeval.py            # dry run: report what it WOULD submit
  tools/autoeval.py --apply    # actually sbatch

Configuration:
  OPD_ROOT              repo root (default: parent of tools/)
  tools/autoeval_arms.json      {"<arm>": {"dir": "<ckpt dir>", "every": 20}, ...}
                        dir may be absolute or relative to OPD_ROOT; it must
                        contain verl-style global_step_<N>/actor/ checkpoints.
  AUTOEVAL_ACCOUNT      sbatch --account   (required for --apply)
  AUTOEVAL_PARTITION    sbatch --partition (required for --apply)
  AUTOEVAL_MAX_INFLIGHT max concurrent autoeval_* jobs (default 3), so
                        auto-eval never crowds out training
  SIF / EXTRA_PKGS      forwarded to scripts/eval/run_eval.sbatch (which needs
                        them; export before running with --apply)
"""
import argparse
import json
import os
import re
import subprocess
import time

OPD = os.environ.get(
    "OPD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
ARMS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autoeval_arms.json")
LEDGER = f"{OPD}/results/autoeval_ledger.json"
STORE = f"{OPD}/results/eval_results.jsonl"

# A checkpoint dir must be untouched this long before we trust it is fully
# written. FSDP writes shards in parallel and a 14-36 GB save takes minutes;
# evaluating a half-written checkpoint wastes a GPU-hour and produces a
# garbage number.
STABLE_SEC = 600
DEFAULT_EVERY = 20  # eval every Nth step -- eval costs ~1 GPU-hour per checkpoint


def load_arms():
    try:
        with open(ARMS_FILE) as f:
            arms = json.load(f)
    except OSError:
        return {}
    for cfg in arms.values():
        if not os.path.isabs(cfg["dir"]):
            cfg["dir"] = os.path.join(OPD, cfg["dir"])
    return arms


def load_ledger():
    try:
        with open(LEDGER) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_ledger(led):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    tmp = LEDGER + ".tmp"
    with open(tmp, "w") as f:
        json.dump(led, f, indent=2, sort_keys=True)
    os.replace(tmp, LEDGER)


def already_graded():
    """(arm, step) pairs that are already in the results store."""
    out = set()
    try:
        with open(STORE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r.get("step"), int):
                    out.add((str(r.get("arm")), r["step"]))
    except OSError:
        pass
    return out


def checkpoint_ready(path):
    """True once every FSDP shard is present and the dir has stopped changing.

    Shards are named model_world_size_<W>_rank_<R>.pt, so W tells us how many
    to expect -- a partially written save is missing ranks, and a save still
    in flight keeps bumping mtimes.
    """
    actor = os.path.join(path, "actor")
    if not os.path.isdir(actor):
        return False, "no actor/"
    try:
        names = os.listdir(actor)
    except OSError:
        return False, "unreadable"
    if "fsdp_config.json" not in names:
        return False, "no fsdp_config.json"
    shards = [n for n in names if n.startswith("model_world_size_")]
    if not shards:
        return False, "no shards"
    m = re.match(r"model_world_size_(\d+)_rank_(\d+)\.pt", shards[0])
    if not m:
        return False, "unparsable shard name"
    world = int(m.group(1))
    ranks = {int(re.search(r"rank_(\d+)", n).group(1)) for n in shards}
    if len(ranks) < world:
        return False, f"{len(ranks)}/{world} shards"
    newest = max(os.path.getmtime(os.path.join(actor, n)) for n in names)
    age = time.time() - newest
    if age < STABLE_SEC:
        return False, f"still settling ({int(age)}s < {STABLE_SEC}s)"
    return True, "ready"


def inflight():
    try:
        out = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 999  # can't tell -> assume busy, submit nothing
    return sum(1 for ln in out.splitlines() if ln.strip().startswith("autoeval_"))


def submit(arm, cfg, step, apply):
    acct = os.environ.get("AUTOEVAL_ACCOUNT")
    part = os.environ.get("AUTOEVAL_PARTITION")
    name = f"autoeval_{arm}_s{step}"
    if not (acct and part):
        return f"NO ACCOUNT/PARTITION for {name}: set AUTOEVAL_ACCOUNT / AUTOEVAL_PARTITION"
    cmd = ["sbatch", "--parsable", f"--job-name={name}",
           f"--account={acct}", f"--partition={part}", "--time=1-00:00:00",
           f"--export=ALL,ARM={arm},CKDIR={cfg['dir']},STEPS={step}",
           f"{OPD}/scripts/eval/run_eval.sbatch"]
    if not apply:
        return f"WOULD SUBMIT {name} on {acct}/{part}"
    try:
        jid = subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return f"SUBMIT FAILED {name}: {e}"
    return f"SUBMITTED {name} -> {jid} on {acct}/{part}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--arm", help="limit to one arm")
    a = ap.parse_args()

    arms = load_arms()
    if not arms:
        print(f"no arms configured -- copy autoeval_arms.example.json to {ARMS_FILE}")
        return

    max_inflight = int(os.environ.get("AUTOEVAL_MAX_INFLIGHT", "3"))
    led = load_ledger()
    graded = already_graded()
    slots = max_inflight - inflight()
    lines = []

    for arm, cfg in sorted(arms.items()):
        if a.arm and arm != a.arm:
            continue
        if not os.path.isdir(cfg["dir"]):
            continue
        every = cfg.get("every", DEFAULT_EVERY)
        for d in sorted(os.listdir(cfg["dir"])):
            m = re.fullmatch(r"global_step_(\d+)", d)
            if not m:
                continue
            step = int(m.group(1))
            if step == 0 or step % every:
                continue
            key = f"{arm}:{step}"
            if key in led or (arm, step) in graded:
                continue
            ok, why = checkpoint_ready(os.path.join(cfg["dir"], d))
            if not ok:
                lines.append(f"  skip {key}: {why}")
                continue
            if slots <= 0:
                lines.append(f"  defer {key}: {max_inflight} auto-evals already in flight")
                continue
            msg = submit(arm, cfg, step, a.apply)
            lines.append("  " + msg)
            if a.apply and msg.startswith("SUBMITTED"):
                led[key] = {"job": msg.split("-> ")[1].split()[0], "at": int(time.time())}
            # decrement in dry-run too, so a dry run previews ONE cycle rather
            # than listing the whole backlog as if it all went out at once
            slots -= 1

    if a.apply:
        save_ledger(led)
    print(f"autoeval ({'apply' if a.apply else 'dry-run'}) — {len(led)} in ledger, "
          f"{max(0, slots)} slot(s) free")
    for ln in lines:
        print(ln)


if __name__ == "__main__":
    main()
