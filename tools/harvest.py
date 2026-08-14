#!/usr/bin/env python3
"""Harvest all graded eval results into one tiny append-only store:
results/eval_results.jsonl. One row =
{domain, arm, step, benchmark, metric, value, n, source}.

Source: scripts/eval/justrl_eval_outputs/<arm>_s<step>/grading_results.json
(what scripts/eval/run_eval.sbatch leaves behind).

Once a row exists here, the checkpoint that produced it is disposable -- the
store is the durable record every table and figure reads from. Idempotent:
re-runs rescan everything and dedupe (later wins).
"""
import glob
import json
import os
import re

OPD = os.environ.get(
    "OPD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
OUT = f"{OPD}/results/eval_results.jsonl"
rows = []

for f in glob.glob(f"{OPD}/scripts/eval/justrl_eval_outputs/*/grading_results.json"):
    tag = os.path.basename(os.path.dirname(f))
    m = re.search(r"^(.*?)[_]?s(?:tep)?[_]?(\d+)$", tag)
    arm, step = (m.group(1), int(m.group(2))) if m else (tag, None)
    try:
        data = json.load(open(f))
    except (OSError, ValueError):
        continue
    for e in (data if isinstance(data, list) else [data]):
        hp = e.get("hyperparameters", {})
        bm = hp.get("task_name", "?")
        for metric in ("mean_score", "best_score", "avg_output_length"):
            if metric in e:
                rows.append(dict(domain="math", arm=arm, step=step, benchmark=bm,
                                 metric=metric, value=e[metric], n=hp.get("n"),
                                 source="grading_results.json"))

# dedupe (later wins) and write
seen = {}
for r in rows:
    seen[(r["domain"], r["arm"], r["step"], r["benchmark"], r["metric"], r["source"])] = r
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    for r in sorted(seen.values(),
                    key=lambda x: (x["domain"], x["arm"], x["step"] or 0,
                                   x["benchmark"], x["metric"])):
        fh.write(json.dumps(r) + "\n")
print(f"WROTE {OUT}: {len(seen)} rows, {os.path.getsize(OUT)/1024:.1f} KB")
print(f"  distinct arms: {len({r['arm'] for r in seen.values()})}")
