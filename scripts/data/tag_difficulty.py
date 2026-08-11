"""Tag DAPO-17k by student pass-rate -> hard subset (see header in prior version).
NOTE: vLLM must be imported and the engine built BEFORE any torch/verl import that
could touch a CUDA context in the parent, or the multiproc workers fail to init.
"""
import os, sys
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def main():

    # --- build the engine FIRST (no CUDA-touching imports before this) ---
    from vllm import LLM, SamplingParams
    MODEL = os.environ["DIFF_MODEL"]
    TP    = int(os.environ.get("DIFF_TP", "4"))
    llm = LLM(model=MODEL, dtype="bfloat16", tensor_parallel_size=TP,
              gpu_memory_utilization=0.85, max_model_len=8192,
              enforce_eager=True, trust_remote_code=True, seed=0)

    # --- now safe to import the rest ---
    import pandas as pd, numpy as np
    sys.path.insert(0, os.environ.get("DIFF_PKGS", "${EXTRA_PKGS}"))
    sys.path.insert(0, "verl")
    from verl.utils.reward_score.ttrl_math import compute_score

    SRC = os.environ.get("DIFF_SRC", "datasets/dapo-math-17k.parquet")
    OUT = os.environ.get("DIFF_OUT", "datasets/dapo_difficulty.parquet")
    N   = int(os.environ.get("DIFF_N", "4"))
    MAXTOK = int(os.environ.get("DIFF_MAXTOK", "4096"))

    df = pd.read_parquet(SRC)
    SUB = int(os.environ.get("DIFF_SUBSAMPLE", "0"))
    if SUB and SUB < len(df):
        df = df.sample(n=SUB, random_state=0).reset_index(drop=True)
    print(f"tagging {len(df)} problems, n={N}", flush=True)
    prompts = [p[0]["content"] if isinstance(p, (list, tuple, np.ndarray)) else p for p in df["prompt"]]
    gts     = [rm["ground_truth"] for rm in df["reward_model"]]

    sp = SamplingParams(n=N, temperature=1.0, top_p=1.0, max_tokens=MAXTOK)
    outs = llm.generate(prompts, sp, use_tqdm=True)
    pr = []
    for out, gt in zip(outs, gts):
        c = sum(1 for o in out.outputs if compute_score(o.text, str(gt))["acc"])
        pr.append(c / N)
    df["pass_rate"] = pr
    df.to_parquet(OUT)
    pr = np.array(pr)
    print(f"pass_rate mean={pr.mean():.3f} ==0:{(pr==0).mean():.3f} <=.25:{(pr<=0.25).mean():.3f}", flush=True)
    for thr, tag in [(0.0, "hard0"), (0.25, "hard25")]:
        h = df[df["pass_rate"] <= thr]
        h.drop(columns=["pass_rate"]).to_parquet(f"datasets/dapo_{tag}.parquet")
        print(f"  {tag}: {len(h)} -> datasets/dapo_{tag}.parquet", flush=True)
    print("TAG_DONE", flush=True)


if __name__ == "__main__":
    main()
