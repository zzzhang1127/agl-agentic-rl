"""Prepared data must be consumed as the same train-derived split."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("data_guard", ROOT / "examples/swe_smith/critic_data_guard.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def fixture(tmp_path):
    split = {"protocol": "critic-train-derived-v1", "heldout_id_disjoint": True, "files": {}}
    smoke = {"instances": {}}
    runtime = {"files": {}}
    for name, iid in (("train.jsonl", "train"), ("validation.jsonl", "validation")):
        row = {"instance_id": iid, "image_name": "source", "problem_statement": iid}
        split["files"][name] = write(tmp_path / "split" / name, row)
        smoke["instances"][name] = [{"instance_id": iid}]
        runtime["files"][name] = write(tmp_path / "smoke_runtime" / name, {**row, "image_name": "runtime"})
    smoke["parent_manifest_sha256"] = write(tmp_path / "split/manifest.json", split)
    runtime["parent_manifest_sha256"] = write(tmp_path / "smoke/manifest.json", smoke)
    write(tmp_path / "smoke_runtime/manifest.json", runtime)
    return tmp_path / "smoke_runtime/train.jsonl", tmp_path / "smoke_runtime/validation.jsonl"


def test_prepared_pair_accepted(tmp_path):
    train, val = fixture(tmp_path)
    guard.validate_critic_smoke_data(str(train), str(val))


@pytest.mark.parametrize("target", ["smoke_runtime/train.jsonl", "split/validation.jsonl", "smoke/manifest.json"])
def test_changed_data_or_provenance_rejected(tmp_path, target):
    train, val = fixture(tmp_path)
    path = tmp_path / target
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="mismatch"):
        guard.validate_critic_smoke_data(str(train), str(val))


def test_heldout_path_cannot_replace_validation(tmp_path):
    train, _ = fixture(tmp_path)
    heldout = tmp_path / "val_dataset_filtered.jsonl"
    heldout.write_text('{"instance_id":"heldout"}\n')
    with pytest.raises(ValueError, match="prepared"):
        guard.validate_critic_smoke_data(str(train), str(heldout))
