from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create an executable three-class smoke dataset from author examples")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo, output = Path(args.repo), Path(args.output)
    source = repo / "1_training" / "Examples" / "training_tumor_benign_SN"
    if output.exists():
        shutil.rmtree(output)
    for class_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        destination = output / class_dir.name
        destination.mkdir(parents=True)
        for image in class_dir.glob("*.jpg"):
            shutil.copy2(image, destination / image.name)
    counts = {path.name: len(list(path.glob("*.jpg"))) for path in output.iterdir()}
    if min(counts.values(), default=0) < 3:
        raise RuntimeError(f"Insufficient example images: {counts}")
    print(counts)


if __name__ == "__main__":
    main()

