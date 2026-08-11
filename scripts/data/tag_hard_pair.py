"""Select the clean hard subset: problems the STUDENT can't solve but the TEACHER can.
Measured at the TRAINING response length (no truncation artifact). vLLM must be built
before any CUDA-touching import; spawn method; run under __main__ guard.
"""
import os, sys
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

def main():
    from vllm import LLM, SamplingParams
    import pandas as pd, numpy as np

    SRC   = os.environ.get("SRC", "datasets/dapo-math-17k.parquet")
    SUB   = int(os.environ.get("SUB", "5000"))
    MAXT  = int(os.environ.get("MAXT", "7168"))
    TP    = int(os.environ.get("TP", "2"))
    NS    = int(os.environ.get("NSTU", "4"))
    NT    = int(os.environ.get("NTCH", "2"))
    STU   = os.environ["STUDENT"]; TCH = os.environ["TEACHER"]

    df = pd.read_parquet(SRC).sample(n=SUB, random_state=0).reset_index(drop=True)
    prompts = [p[0]["content"] if isinstance(p,(list,tuple,np.ndarray)) else p for p in df["prompt"]]
    gts = [rm["ground_truth"] for rm in df["reward_model"]]
    print(f"pair-tagging {len(df)} problems @ {MAXT} tok, student n={NS} teacher n={NT}", flush=True)

    def passrate(model, n):
        llm = LLM(model=model, dtype="bfloat16", tensor_parallel_size=TP,
                  gpu_memory_utilization=0.85, max_model_len=MAXT+1024,
                  enforce_eager=True, trust_remote_code=True, seed=0)
        outs = llm.generate(prompts, SamplingParams(n=n, temperature=1.0, top_p=1.0, max_tokens=MAXT), use_tqdm=True)
        sys.path.insert(0, "${EXTRA_PKGS}"); sys.path.insert(0, "verl")
        from verl.utils.reward_score.ttrl_math import compute_score
        pr = [sum(1 for o in out.outputs if compute_score(o.text,str(gt))["acc"])/n for out,gt in zip(outs,gts)]
        import gc, torch
        del llm; gc.collect(); torch.cuda.empty_cache()
        return np.array(pr)

    s_pr = passrate(STU, NS); print(f"student pass: mean={s_pr.mean():.3f} ==0:{(s_pr==0).mean():.3f}", flush=True)
    t_pr = passrate(TCH, NT); print(f"teacher pass: mean={t_pr.mean():.3f} >=.5:{(t_pr>=0.5).mean():.3f}", flush=True)
    df["student_pass"] = s_pr; df["teacher_pass"] = t_pr
    df.to_parquet("datasets/dapo_pairtag.parquet")

    for sthr, tthr, tag in [(0.0, 0.5, "hardgap"), (0.0, 0.25, "hardgap_lo"), (0.25, 0.5, "hardgap_wide")]:
        m = (df["student_pass"] <= sthr) & (df["teacher_pass"] >= tthr)
        sub = df[m].drop(columns=["student_pass","teacher_pass"])
        sub.to_parquet(f"datasets/dapo_{tag}.parquet")
        print(f"  {tag} (stu<={sthr}, tch>={tthr}): {len(sub)} -> datasets/dapo_{tag}.parquet", flush=True)
    print("PAIRTAG_DONE", flush=True)

if __name__ == "__main__":
    main()
