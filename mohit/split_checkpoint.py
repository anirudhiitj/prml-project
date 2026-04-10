"""
split_checkpoint.py — Split large .pt checkpoint into <100 MB parts for GitHub.

Usage:
    # Split (default 2 parts for 150 MB file → ~75 MB each)
    python split_checkpoint.py --input checkpoints/best_model.pt

    # Merge parts back
    python split_checkpoint.py --merge checkpoints/best_model.pt
"""

import argparse
import math
from pathlib import Path


def split(input_path: str, n_parts: int = 2) -> None:
    src = Path(input_path)
    if not src.exists():
        print(f"File not found: {src}")
        return

    total_size = src.stat().st_size
    part_size = math.ceil(total_size / n_parts)

    print(f"Splitting: {src.name}")
    print(f"  Total size : {total_size / 1024**2:.1f} MB")
    print(f"  Parts      : {n_parts}")
    print(f"  Part size  : ~{part_size / 1024**2:.1f} MB each")

    with open(src, "rb") as f:
        for i in range(n_parts):
            chunk = f.read(part_size)
            if not chunk:
                break
            part_path = Path(str(src) + f".part{i}")
            part_path.write_bytes(chunk)
            print(f"  {part_path.name}  ({len(chunk) / 1024**2:.1f} MB)")

    print(f"\nDone. Commit the .part* files to GitHub.")


def merge(checkpoint_path: str) -> bool:
    """Merge .part* files back into a single .pt. Returns True if merge happened."""
    target = Path(checkpoint_path)

    parts = sorted(target.parent.glob(target.name + ".part*"),
                   key=lambda p: int(p.suffix.lstrip(".part")))

    if not parts:
        return False

    if target.exists():
        print(f"  {target.name} already exists, skipping merge.")
        return False

    print(f"Merging {len(parts)} parts -> {target.name} ...")
    with open(target, "wb") as out:
        for part in parts:
            out.write(part.read_bytes())
            print(f"  + {part.name}")

    size_mb = target.stat().st_size / 1024**2
    print(f"  Merged: {target.name} ({size_mb:.1f} MB)")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split/merge checkpoint files")
    parser.add_argument("--input", type=str, help="Path to .pt file to split")
    parser.add_argument("--merge", type=str, help="Path to .pt file to merge from parts")
    parser.add_argument("--parts", type=int, default=2, help="Number of parts (default: 2)")
    args = parser.parse_args()

    if args.merge:
        merge(args.merge)
    elif args.input:
        split(args.input, args.parts)
    else:
        parser.print_help()
