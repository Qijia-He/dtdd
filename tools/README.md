# tools/ — optional SLURM automation

Nothing in the training or eval pipeline depends on this directory; it is the
operational layer we ran the paper's experiment campaign with. Two of the
files are generic and runnable against this repo as-is; two are shipped as
annotated site-specific references.

## Runnable

| file | what it does |
|---|---|
| `autoeval.py` | **Checkpoint saved ⇒ eval submitted.** Scans configured checkpoint dirs (`autoeval_arms.json`, see the `.example.json`); when a `global_step_N/` is shard-complete and has stopped changing for 10 min, submits `scripts/eval/run_eval.sbatch` for it. Idempotent via a ledger; throttled so eval never crowds out training. Dry-run by default, `--apply` to submit. |
| `harvest.py` | Folds every `grading_results.json` the eval pipeline produced into one small append-only store, `results/eval_results.jsonl`. Once a result row exists, the checkpoint that produced it is disposable. |

Why this exists: every coverage hole in our experiment record came from
"train now, evaluate later" — checkpoints that were never graded, or were
pruned in a disk cleanup before grading. Evaluating at save time makes the
*result* durable independently of the checkpoint (the store is kilobytes;
checkpoints are terabytes).

Typical loop (cron or a monitor daemon):

```bash
export OPD_ROOT=/path/to/opd SIF=/path/to/verl_vllm011.sif \
       AUTOEVAL_ACCOUNT=myacct AUTOEVAL_PARTITION=gpu-a100
cp tools/autoeval_arms.example.json tools/autoeval_arms.json  # edit arms
python3 tools/autoeval.py --apply && python3 tools/harvest.py
```

## Reference (site-specific, adapt before use)

| file | what it demonstrates |
|---|---|
| `monitor_daemon.sh` | The persistent monitor that babysat the campaign: stall detection via newest-log-mtime (SLURM `RUNNING` ≠ alive), FASTKILL on fatal signals, singleton guard + self-renew on short-walltime queues, staggered auto-resumes, verifying a checkpoint *loads* before resuming from it, save⇒eval⇒harvest wiring, and disk **and inode** quota sentinels. Paths/partitions/arm tables are from our cluster (UW Hyak) and its second, agentic experiment lane — edit before running. |
| `track.py` | One-screen dashboard: live jobs with current training step, jobs that *ended without their success marker* (plain `COMPLETED` hides those), and the results store. Edit the `KNOWN` table for your runs. |
