"""Build the math training set: datasets/dapo-math-17k.parquet.

The Hugging Face release of DAPO-Math-17K (BytedTsinghua-SIA/DAPO-Math-17k)
stores each of its 17,917 problems 100 times (1,791,700 rows). Keeping the
first row per problem id reproduces, row for row and column for column, the
parquet every math number in the paper was trained on. (Deduplicating by
prompt text instead would give 17,398 rows: a few distinct problems share a
prompt.)

    python math/prepare_data.py            # writes $DTDD_ROOT/datasets/dapo-math-17k.parquet

The evaluation sets (AIME24, AIME25, AMC23) need no preparation: setup.sh links
them from the Thinking-Space/Rethinking-OPD checkout.
"""
import os

import pandas as pd
from huggingface_hub import hf_hub_download

root = os.environ.get("DTDD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out = os.path.join(root, "datasets", "dapo-math-17k.parquet")

raw = hf_hub_download("BytedTsinghua-SIA/DAPO-Math-17k", "data/dapo-math-17k.parquet", repo_type="dataset")
df = pd.read_parquet(raw)
ids = df["extra_info"].map(lambda e: e["index"])
df = df[~ids.duplicated(keep="first")].reset_index(drop=True)
assert len(df) == 17917, len(df)

os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_parquet(out)
print(f"{len(df)} problems -> {out}")
