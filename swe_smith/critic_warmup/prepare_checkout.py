"""Validate/apply the critic lineage to a clean, dedicated upstream checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Apply after all preflight checks; default only checks")
    args = parser.parse_args()
    root = args.checkout.resolve()
    package = Path(__file__).resolve().parent
    manifest = json.loads((package / "overlay_manifest.json").read_text())
    patch = package.parents[1] / "patches/critic-warmup-20261015.diff"

    def git(*argv):
        return subprocess.check_output(["git", "-C", str(root), *argv], text=True).strip()

    if git("rev-parse", "HEAD") != manifest["upstream_base"]:
        parser.error("Use a dedicated checkout at the exact upstream_base in overlay_manifest.json")
    if git("status", "--porcelain"):
        parser.error("Checkout must be clean; preserve existing work in a different checkout")
    seen = set()
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or item["path"] in seen:
            parser.error("invalid or duplicate overlay path")
        seen.add(item["path"])
        source = package / "overlay" / relative
        target = root / relative
        if not source.resolve().is_relative_to(package / "overlay") or not target.resolve().is_relative_to(root):
            parser.error("overlay path escapes package or checkout")
        if hashlib.sha256(source.read_bytes()).hexdigest() != item["sha256"]:
            parser.error("overlay checksum mismatch: " + item["path"])
    subprocess.run(["git", "-C", str(root), "apply", "--check", str(patch)], check=True)
    if args.apply:
        subprocess.run(["git", "-C", str(root), "apply", str(patch)], check=True)
        for item in manifest["files"]:
            target = root / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(package / "overlay" / item["path"], target)
    print(json.dumps({"base": manifest["upstream_base"], "overlay_files": len(seen), "applied": args.apply}))


if __name__ == "__main__":
    main()
