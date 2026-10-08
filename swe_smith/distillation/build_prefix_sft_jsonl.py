#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

"""Write MiniCPM SFT jsonl from shortest-prefix trajectory rows.

Each kept row is already ``resolved`` with truncated ``messages``. This does not
hard-cap turns (no short15). Init checkpoint for the actual SFT job remains ep3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def keep_row(row: dict[str, Any]) -> bool:
    if row.get("drop"):
        return False
    if not row.get("resolved"):
        return False
    messages = row.get("messages")
    return isinstance(messages, list) and bool(messages)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="src", required=True, help="jsonl of shortest-prefix rows")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    n_in = n_out = 0
    with Path(args.src).open(encoding="utf-8") as src, Path(args.out).open("w", encoding="utf-8") as out:
        for line in src:
            if not line.strip():
                continue
            n_in += 1
            row = json.loads(line)
            if not keep_row(row):
                continue
            n_out += 1
            out.write(
                json.dumps(
                    {
                        "instance_id": row.get("instance_id"),
                        "messages": row["messages"],
                        "n_turns": row.get("n_turns"),
                        "n_turns_original": row.get("n_turns_original"),
                        "resolved": True,
                        "submitted": True,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(json.dumps({"in": n_in, "out": n_out, "path": args.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
