import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from modern_pca import reimplementation as r


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()

    if output.exists():
        raise FileExistsError(f"Cache directory already exists: {output}")

    manifest = root / "manifests/singapore_epoch15_all.csv"

    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 11020
    assert len({row["sample_id"] for row in rows}) == 11020
    assert all(row["label"] in ("0", "1") for row in rows)

    if args.limit:
        assert args.limit > 0
        rows = rows[:args.limit]

    output.mkdir(parents=True)

    images = open_memmap(
        output / "images.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), 350, 350, 3),
    )

    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    np.save(output / "labels.npy", labels)

    normalizers = r.normalizers(r.REFERENCE)

    started = time.perf_counter()
    metadata = []

    for index, row in enumerate(rows):
        path = (root / row["image_path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Invalid image path: {path}")

        file_hash = sha256(path)
        checked = {
            "sample_id": row["sample_id"],
            "file_sha256": file_hash,
        }

        images[index] = r.preprocess_legacy(
            checked, path, normalizers
        )

        metadata.append({
            **row,
            "file_sha256": file_hash,
        })

        count = index + 1
        if count % 100 == 0 or count == len(rows):
            images.flush()
            elapsed = time.perf_counter() - started
            print(
                f"CACHE {count}/{len(rows)} "
                f"elapsed_min={elapsed / 60:.2f}",
                flush=True,
            )

    images.flush()
    del images

    with (output / "samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)

    verification = {
        "status": "PASS",
        "samples": len(rows),
        "shape": [len(rows), 350, 350, 3],
        "dtype": "float32",
        "labels": {
            "benign": int(np.sum(labels == 0)),
            "tumour": int(np.sum(labels == 1)),
        },
        "source_manifest_sha256": sha256(manifest),
        "images_sha256": sha256(output / "images.npy"),
        "labels_sha256": sha256(output / "labels.npy"),
        "samples_sha256": sha256(output / "samples.csv"),
        "preprocessing": "legacy Lanczos + brightness + Macenko + float32/255",
        "reference_sha256": r.REFERENCE_SHA,
        "elapsed_seconds": time.perf_counter() - started,
    }

    (output / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )

    print("PASS: Cache built and fingerprinted", flush=True)


if __name__ == "__main__":
    main()
