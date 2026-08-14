#!/usr/bin/env python3
"""Live one-screen tracker for all running experiments.

SITE-SPECIFIC REFERENCE: this is the dashboard we ran during the 2026-07/08
campaign, spanning this repo's math lane and a separate agentic lane (the
"agent"/"bench" entries in KNOWN). Edit KNOWN and the env paths for your own
runs; the useful ideas are (a) job -> current-step resolution from run logs,
(b) surfacing jobs that ENDED without their success marker -- SLURM's COMPLETED
state alone hides those.

    python3 tools/track.py           # one-shot dashboard
    python3 tools/track.py --watch N # refresh every N seconds
Reads squeue + run logs + the results store. No side effects.
"""
import subprocess, re, os, json, glob, sys, time

OPD  = os.environ.get("OPD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AG   = os.environ.get("AGENT_CKPTS", "")          # agentic lane, optional
CSEL = os.environ.get("SCRATCH2_LOGS", "")        # second log dir, optional
MLOG = f"{OPD}/logs"
STORE= f"{OPD}/results/eval_results.jsonl"
USER = os.environ.get("USER", "")

# job-name prefix -> (label, kind, where progress lives)   -- EDIT ME
KNOWN = [
 ("dtddA_taufix",  "MATH  cold-start DTDD (tau=0.2)",      "math",  f"{MLOG}/dtddA_taufix_*.out"),
 ("dtddB",         "MATH  length-sweep DTDD arm",          "math",  f"{MLOG}/dtddB_*.out"),
 ("autoeval_",     "EVAL  auto-submitted held-out eval",   "misc",  None),
]

def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=25).stdout
    except Exception: return ""

def math_step(pat):
    fs=sorted(glob.glob(pat), key=os.path.getmtime)
    if not fs: return None
    s=re.findall(r"training/global_step:(\d+)", open(fs[-1],errors="ignore").read())
    return int(s[-1]) if s else 0

def classify(name):
    for pre,label,kind,src in KNOWN:
        if name.startswith(pre): return label,kind,src
    return name,"?",None

def once():
    print("="*86)
    print(f"  EXPERIMENT TRACKER   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*86)

    # ---------------- live jobs ----------------
    q=sh(f"squeue -u {USER} -h -o '%i|%j|%T|%M|%R'").strip().splitlines()
    print("\n### LIVE JOBS")
    if not q: print("  (none)")
    for line in q:
        jid,name,state,tm,reason = (line.split("|")+[""]*5)[:5]
        if "Reservation" in reason: continue
        label,kind,src = classify(name.strip())
        extra=""
        if kind=="math":
            st=math_step(src);  extra=f"step {st}" if st is not None else "starting"
        print(f"  {jid.strip():>9} {state.strip():<9} {tm.strip():>9}  {label:<38} {extra}")

    # ---------------- recent failures ----------------
    print("\n### ENDED IN LAST 12h WITHOUT SUCCESS MARKER")
    for line in sh(f"sacct -u {USER} --starttime now-12hours -X -n -o JobID,JobName%20,State,Elapsed").strip().splitlines():
        p=line.split()
        if len(p)<4: continue
        jid,name,state,el=p[0],p[1],p[2],p[3]
        if state in ("RUNNING","PENDING"): continue
        o=glob.glob(f"{MLOG}/{name}_{jid}.out") + (glob.glob(f"{CSEL}/{name}_{jid}.out") if CSEL else [])
        txt=open(o[0],errors="ignore").read() if o else ""
        ok_marker=any(k in txt for k in ("EVAL_DONE","SMOKE_OK","Training completed"))
        h,m,s=(el.split("-")[-1].split(":")+["0","0"])[:3]
        quick=(int(h)*60+int(m))<10
        if state.startswith("CANCELLED") or (state=="COMPLETED" and ok_marker):
            continue                                 # deliberate kill or verified success
        if state!="COMPLETED" or quick:
            why="FAILED" if state!="COMPLETED" else "ended fast w/o success marker"
            print(f"  {jid:>9} {state:<11} {el:>10}  {name:<22} <-- {why}")

    # ---------------- results ----------------
    print("\n### HELD-OUT RESULTS (results store)")
    if os.path.exists(STORE):
        d={}
        for l in open(STORE):
            r=json.loads(l)
            if r.get("metric")=="mean_score":
                d.setdefault((r["arm"],r["step"]),{})[r["benchmark"]]=r["value"]
        for (arm,step),v in sorted(d.items()):
            print(f"  {arm:>12} s{step:<5} " + "  ".join(f"{k}={x:.3f}" for k,x in sorted(v.items())))
    else:
        print("  (no store yet — run tools/harvest.py)")
    print("\n(refresh results: python3 tools/harvest.py)")

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--watch":
        while True:
            once(); time.sleep(int(sys.argv[2]))
    else:
        once()
