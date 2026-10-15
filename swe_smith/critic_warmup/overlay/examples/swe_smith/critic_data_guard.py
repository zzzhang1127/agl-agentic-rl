# Copyright (c) Microsoft. All rights reserved.

"""Verify the prepared critic smoke split before loading any training workers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def validate_critic_smoke_data(train_path: str, validation_path: str) -> None:
    train = Path(train_path).resolve(strict=True)
    validation = Path(validation_path).resolve(strict=True)
    if train.parent != validation.parent or train.name != "train.jsonl" or validation.name != "validation.jsonl":
        raise ValueError("critic smoke requires the prepared train.jsonl/validation.jsonl pair")
    root = train.parent
    runtime = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    smoke_path = root.parent / "smoke/manifest.json"
    split_path = root.parent / "split/manifest.json"
    smoke_bytes, split_bytes = smoke_path.read_bytes(), split_path.read_bytes()
    if hashlib.sha256(smoke_bytes).hexdigest() != runtime.get("parent_manifest_sha256"):
        raise ValueError("critic smoke runtime provenance mismatch")
    smoke = json.loads(smoke_bytes)
    split = json.loads(split_bytes)
    if hashlib.sha256(split_bytes).hexdigest() != smoke.get("parent_manifest_sha256"):
        raise ValueError("critic smoke split provenance mismatch")
    if split.get("protocol") != "critic-train-derived-v1" or split.get("heldout_id_disjoint") is not True:
        raise ValueError("critic smoke requires a train-derived, heldout-disjoint split")
    selected_ids = []
    for path in (train, validation):
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != runtime.get("files", {}).get(path.name):
            raise ValueError("critic smoke data hash mismatch")
        parent_data = (split_path.parent / path.name).read_bytes()
        if hashlib.sha256(parent_data).hexdigest() != split.get("files", {}).get(path.name):
            raise ValueError("critic parent split data hash mismatch")
        parents = {row["instance_id"]: row for row in map(json.loads, parent_data.splitlines())}
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        ids = [row["instance_id"] for row in rows]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("empty or duplicate critic smoke IDs")
        expected = [row["instance_id"] for row in smoke["instances"][path.name]]
        if ids != expected:
            raise ValueError("critic smoke membership changed")
        for row in rows:
            original = parents.get(row["instance_id"])
            if original is None or {k: v for k, v in row.items() if k != "image_name"} != {
                k: v for k, v in original.items() if k != "image_name"
            }:
                raise ValueError("critic smoke task differs from parent split")
        selected_ids.append(set(ids))
    if selected_ids[0] & selected_ids[1]:
        raise ValueError("critic train/validation overlap")
