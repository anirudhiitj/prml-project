"""
split_checkpoints.py — Split large .pt checkpoint files into smaller parts
for committing to GitHub (100 MB per file limit).

Usage:
    # Split into 4 parts (default)
    python split_checkpoints.py --input anirudh/cocktail_separation/checkpoints/best2.pt

    # Split into N parts
    python split_checkpoints.py --input path/to/model.pt --parts 5

    # Merge parts back manually (auto-merge also happens at inference time)
    python split_checkpoints.py --merge path/to/model.pt

The parts are saved alongside the original:
    best2.pt  →  best2.pt.part0  best2.pt.part1  best2.pt.part2  best2.pt.part3
"""

import argparse
import os
import math
from pathlib import Path


def split(input_path: str, n_parts: int = 4) -> None:
    src = Path(input_path)
    if not src.exists():
        print(f"❌ File not found: {src}")
        return

    total_size = src.stat().st_size
    part_size = math.ceil(total_size / n_parts)

    print(f"📦 Splitting: {src.name}")
    print(f"   Total size : {total_size / 1024**2:.1f} MB")
    print(f"   Parts      : {n_parts}")
    print(f"   Part size  : ~{part_size / 1024**2:.1f} MB each")

    with open(src, "rb") as f:
        for i in range(n_parts):
            chunk = f.read(part_size)
            if not chunk:
                break
            part_path = Path(str(src) + f".part{i}")
            part_path.write_bytes(chunk)
            print(f"   ✅ {part_path.name}  ({len(chunk) / 1024**2:.1f} MB)")

    print(f"\n✔  Done. Commit the .part* files. The original .pt is in .gitignore.")


def merge(checkpoint_path: str) -> bool:
    """
    Merge .part* files back into a single .pt.
    Returns True if merge happened, False if already merged or no parts found.
    """
    target = Path(checkpoint_path)

    # Collect all parts in order
    parts = sorted(target.parent.glob(target.name + ".part*"),
                   key=lambda p: int(p.suffix.lstrip(".part")))

    if not parts:
        return False

    if target.exists():
        print(f"   ℹ️  {target.name} already exists, skipping merge.")
        return False

    print(f"🔧 Merging {len(parts)} parts → {target.name} ...")
    with open(target, "wb") as out:
        for part in parts:
            out.write(part.read_bytes())
            print(f"   + {part.name}")

    size_mb = target.stat().st_size / 1024**2
    print(f"   ✅ Merged: {target.name} ({size_mb:.1f} MB)")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split or merge .pt checkpoint files")
    parser.add_argument("--input", type=str, help="Path to .pt file to split")
    parser.add_argument("--parts", type=int, default=4, help="Number of parts (default: 4)")
    parser.add_argument("--merge", type=str, help="Path to .pt to reassemble from parts")
    args = parser.parse_args()

    if args.input:
        split(args.input, args.parts)
    elif args.merge:
        merged = merge(args.merge)
        if not merged:
            print("Nothing to merge.")
    else:
        parser.print_help()
