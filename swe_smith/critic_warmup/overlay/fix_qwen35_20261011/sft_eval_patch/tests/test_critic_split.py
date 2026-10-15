"""Train-derived split invariants, independent of training dependencies."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("critic_split", ROOT / "examples/swe_smith/prepare_critic_split.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sources(tmp_path):
    rows = [{"instance_id": f"i{i}", "problem_statement": f"fix task {i}", "image_name": "image",
             "repo": "repo", "FAIL_TO_PASS": ["test.py::test_x"], "PASS_TO_PASS": [],
             "patch": "MUST NOT EXPORT"} for i in range(100)]
    rows.append({**rows[0], "instance_id": "same-prompt", "problem_statement": " fix   task 0 "})
    train, heldout = tmp_path / "source.jsonl", tmp_path / "heldout.jsonl"
    train.write_text("".join(json.dumps(r) + "\n" for r in rows))
    heldout.write_text(json.dumps({"instance_id": "heldout-only"}) + "\n")
    return train, heldout


def test_reproducible_disjoint_and_projected(tmp_path):
    train, heldout = sources(tmp_path)
    first, second = tmp_path / "one", tmp_path / "two"
    a = module.prepare(train, heldout, first)
    b = module.prepare(train, heldout, second)
    assert a == b
    sets = []
    for name in ("train.jsonl", "validation.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
        rows = [json.loads(line) for line in (first / name).read_text().splitlines()]
        assert all(set(row) == set(module.FIELDS) for row in rows)
        ids = {row["instance_id"] for row in rows}
        assert ("i0" in ids) == ("same-prompt" in ids)
        sets.append(ids)
    assert not sets[0] & sets[1]
    assert len(sets[0] | sets[1]) == 101
    with pytest.raises(FileExistsError):
        module.prepare(train, heldout, first)


@pytest.mark.parametrize("bad", ["overlap", "duplicate"])
def test_invalid_sources_do_not_write_output(tmp_path, bad):
    train, heldout = sources(tmp_path)
    if bad == "overlap":
        heldout.write_text(json.dumps({"instance_id": "i0"}) + "\n")
    else:
        with train.open("a") as file:
            file.write(train.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError):
        module.prepare(train, heldout, tmp_path / "output")
    assert not (tmp_path / "output").exists()
