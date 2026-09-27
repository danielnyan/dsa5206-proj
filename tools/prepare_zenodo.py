from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description="Index extracted Zenodo validation archives by binary class")
    parser.add_argument("--archives-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.archives_root), Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    counts = {"benign": 0, "tumor": 0}
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUFFIXES:
            continue
        lower = str(path).lower()
        class_name = "tumor" if "_tu" in lower or "/tu" in lower or "tumor" in lower else "benign"
        destination = output / class_name
        destination.mkdir(parents=True, exist_ok=True)
        unique = f"{path.parent.name}__{path.name}"
        target = destination / unique
        if target.exists():
            unique = f"{counts[class_name]:09d}__{unique}"
            target = destination / unique
        os.symlink(path.resolve(), target)
        counts[class_name] += 1
    if min(counts.values()) == 0:
        raise RuntimeError(f"Could not identify both classes: {counts}")
    print(counts)


if __name__ == "__main__":
    main()

