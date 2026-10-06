#!/usr/bin/env python3
"""Group SWE-smith rows by image_name so one repo image can serve many containers."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if not rows:
        raise SystemExit(f"empty dataset: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--batch-images", type=int, default=8, help="images per pull batch")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.datasets:
        rows.extend(load_jsonl(path))

    by_image: dict[str, list[str]] = defaultdict(list)
    by_repo: Counter[str] = Counter()
    missing = 0
    for row in rows:
        image = row.get("image_name") or ""
        repo = row.get("repo") or ""
        instance = str(row.get("instance_id") or "")
        if not image:
            missing += 1
            continue
        by_image[image].append(instance)
        by_repo[repo or image] += 1

    images = sorted(by_image, key=lambda name: (-len(by_image[name]), name))
    print(f"rows={len(rows)} missing_image_name={missing} distinct_images={len(images)} distinct_repos={len(by_repo)}")
    print("Same image_name = same Docker layers. Concurrent containers of one image are overlay copy-on-write,")
    print("so 4 jobs on repo X cost about 1 image + small writable layers, not 4x the image.")
    print()
    print("Top images by instance count:")
    for image in images[:20]:
        print(f"  {len(by_image[image]):5d}  {image}")

    print()
    print(f"Suggested pull batches of {args.batch_images} images (do not pull into /var/lib/docker on this host):")
    for start in range(0, len(images), args.batch_images):
        chunk = images[start : start + args.batch_images]
        n = sum(len(by_image[name]) for name in chunk)
        print(f"  batch {start // args.batch_images + 1:02d}: {len(chunk)} images, {n} instances")
        for name in chunk:
            print(f"    {name}")


if __name__ == "__main__":
    main()
