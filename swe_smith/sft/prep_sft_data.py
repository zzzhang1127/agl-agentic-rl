"""Clean teacher trajectories (resolved && no flags) -> parquet for SFT, with a small held-out val split
(split by problem so segments of one rollout never straddle train/val)."""
import json, random, sys
import pandas as pd

src = sys.argv[1]
out_dir = sys.argv[2]
rows = [json.loads(l) for l in open(src)]
ids = sorted({r["instance_id"] for r in rows})
rng = random.Random(20260920)
val_ids = set(rng.sample(ids, 30))
recs = []
for r in rows:
    recs.append(dict(
        instance_id=r["instance_id"], run=r["run"], segment=r["segment"], segment_index=r["segment_index"],
        messages_json=json.dumps(r["messages"], ensure_ascii=False),
        tools_json=json.dumps(r["tool_schemas"], ensure_ascii=False),
        split="val" if r["instance_id"] in val_ids else "train",
    ))
df = pd.DataFrame(recs)
tr = df[df.split == "train"].sample(frac=1.0, random_state=1).reset_index(drop=True)
va = df[df.split == "val"].reset_index(drop=True)
tr.to_parquet(f"{out_dir}/train.parquet", index=False)
va.to_parquet(f"{out_dir}/val.parquet", index=False)
print("train", len(tr), tr.segment.value_counts().to_dict(), "val", len(va), va.segment.value_counts().to_dict())
