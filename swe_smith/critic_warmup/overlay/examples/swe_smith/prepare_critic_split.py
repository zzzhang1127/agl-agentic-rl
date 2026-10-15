# Copyright (c) Microsoft. All rights reserved.

"""Create deterministic train-derived critic splits without reading reference patches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FIELDS = ("instance_id", "problem_statement", "image_name", "repo", "FAIL_TO_PASS", "PASS_TO_PASS")


def prepare(train_path: Path, heldout_path: Path, output: Path) -> dict:
    train_bytes, heldout_bytes = train_path.read_bytes(), heldout_path.read_bytes()
    heldout_ids = {json.loads(line)["instance_id"] for line in heldout_bytes.splitlines() if line.strip()}
    seen, groups = set(), {}
    for line in train_bytes.splitlines():
        if not line.strip():
            continue
        source = json.loads(line)
        row = {key: source[key] for key in FIELDS}
        iid = row["instance_id"]
        if not isinstance(iid, str) or not iid or iid in seen:
            raise ValueError("missing or duplicate training instance ID")
        if iid in heldout_ids:
            raise ValueError("training source overlaps heldout instance IDs")
        seen.add(iid)
        if not isinstance(row["problem_statement"], str) or not row["problem_statement"].strip():
            raise ValueError("missing problem statement")
        # Keep identical normalized task prompts together across the two splits.
        prompt = " ".join(row["problem_statement"].split())
        group = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        groups.setdefault(group, []).append(row)
    train, validation = [], []
    for group, rows in sorted(groups.items()):
        # Fixed hash partition; neither model scores nor heldout outcomes select rows.
        bucket = int(hashlib.sha256(("critic-v1:" + group).encode()).hexdigest(), 16) % 10
        destination = validation if bucket == 0 else train
        destination.extend(sorted(rows, key=lambda row: row["instance_id"]))
    if not train or not validation:
        raise ValueError("hash partition produced an empty split")
    manifest = {
        "protocol": "critic-train-derived-v1", "validation_hash_buckets": [0], "hash_bucket_count": 10,
        "train_source": str(train_path.resolve()), "train_source_sha256": hashlib.sha256(train_bytes).hexdigest(),
        "heldout_id_source": str(heldout_path.resolve()), "heldout_source_sha256": hashlib.sha256(heldout_bytes).hexdigest(),
        "source_count": len(seen), "prompt_groups": len(groups), "heldout_id_disjoint": True,
        "train_count": len(train), "validation_count": len(validation), "files": {},
    }
    payloads = {}
    for name, rows in (("train.jsonl", train), ("validation.jsonl", validation)):
        data = ("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)).encode("utf-8")
        payloads[name] = data
        manifest["files"][name] = hashlib.sha256(data).hexdigest()
    # Never replace an existing split or silently reuse a partially written one.
    output.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        with (output / name).open("xb") as file:
            file.write(data)
    with (output / "manifest.json").open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", required=True, type=Path)
    parser.add_argument("--heldout-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.train_source, args.heldout_source, args.output), indent=2))
