#!/usr/bin/env python3
"""Prepare one PBS-ready PANDA Gleason directory directly from chunk ZIPs.

Expected input directory::

    panda_150um_chunk_0000.zip
    panda_150um_chunk_0000.zip.sha256.txt
    panda_150um_chunk_0001.zip
    panda_150um_chunk_0001.zip.sha256.txt
    ...

The ZIPs are the transfer shards produced by the Kaggle chunk exporter. They
are never combined into another archive. This helper verifies every archive
SHA-256 sidecar, validates cross-shard metadata, streams the selected PNGs
straight from each ZIP into one prepared cache, and writes the manifest/cache
contract consumed by ``modern_pca.train_gleason``.

By default a complete contiguous cohort is required. ``--allow-partial`` is
explicitly for preliminary/smoke experiments and marks ``sample_only=true``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

CLASSES = ["GP3", "GP4", "GP5"]
ARCHIVE_RE = re.compile(r"^panda_150um_chunk_(\d{4})\.zip$")


def sha256_file(path: Path, block: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_sidecar(sidecar: Path, expected_name: str) -> str:
    if not sidecar.is_file():
        raise FileNotFoundError(
            f"Missing SHA-256 sidecar for {expected_name}: {sidecar}"
        )
    line = sidecar.read_text(encoding="utf-8").strip()
    parts = line.split()
    if len(parts) < 1 or re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]) is None:
        raise ValueError(f"Malformed SHA-256 sidecar: {sidecar}")
    if len(parts) >= 2 and Path(parts[-1]).name != expected_name:
        raise ValueError(
            f"Sidecar filename mismatch: {sidecar} names {parts[-1]!r}, "
            f"expected {expected_name!r}"
        )
    return parts[0].lower()


def safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(p in ("", ".", "..") for p in path.parts):
        raise ValueError(f"Unsafe ZIP member path: {name!r}")
    return path


def is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


def read_member(zf: zipfile.ZipFile, member: str) -> bytes:
    info = zf.getinfo(member)
    safe_member(info.filename)
    if info.is_dir() or is_symlink(info):
        raise ValueError(f"Expected regular ZIP file member: {member}")
    return zf.read(info)


def read_csv_bytes(payload: bytes) -> tuple[list[str], list[dict]]:
    text = payload.decode("utf-8")
    stream = io.StringIO(text, newline="")
    reader = csv.DictReader(stream)
    rows = list(reader)
    if not reader.fieldnames:
        raise ValueError("Manifest has no CSV header")
    return list(reader.fieldnames), rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stream_cache_member(
    zf: zipfile.ZipFile,
    member: str,
    target: Path,
    expected_sha: str,
) -> None:
    info = zf.getinfo(member)
    safe_member(info.filename)
    if info.is_dir() or is_symlink(info):
        raise ValueError(f"Unsafe/non-file cache member: {member}")

    if target.exists():
        if sha256_file(target) != expected_sha:
            raise ValueError(f"Cache collision with different content: {target.name}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()

    with zf.open(info, "r") as source, temporary.open("wb") as sink:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            sink.write(block)

    actual = digest.hexdigest()
    if actual != expected_sha:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"Cache SHA-256 mismatch for {target.name}: {actual} != {expected_sha}"
        )
    temporary.replace(target)


def discover_archives(chunks_dir: Path) -> list[tuple[int, Path, str]]:
    chunks_dir = Path(chunks_dir).resolve(strict=True)
    found = []
    for path in sorted(chunks_dir.iterdir()):
        if not path.is_file():
            continue
        match = ARCHIVE_RE.fullmatch(path.name)
        if not match:
            continue
        index = int(match.group(1))
        sidecar = path.with_suffix(path.suffix + ".sha256.txt")
        expected = parse_sidecar(sidecar, path.name)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Archive SHA-256 mismatch: {path.name}\n"
                f"expected {expected}\nactual   {actual}"
            )
        found.append((index, path, actual))
    if not found:
        raise FileNotFoundError(
            f"No panda_150um_chunk_XXXX.zip archives found in {chunks_dir}"
        )
    indices = [x[0] for x in found]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate chunk indices: {indices}")
    return sorted(found)


def open_chunk(index: int, path: Path, archive_sha: str) -> dict:
    zf = zipfile.ZipFile(path, "r")
    bad = zf.testzip()
    if bad is not None:
        zf.close()
        raise ValueError(f"ZIP CRC failure in {path.name}: {bad}")

    prefixes = {
        PurePosixPath(info.filename).parts[0]
        for info in zf.infolist()
        if info.filename and not info.is_dir()
    }
    expected_prefix = f"panda-150um-chunk_{index:04d}"
    if prefixes != {expected_prefix}:
        zf.close()
        raise ValueError(
            f"{path.name} must contain exactly top-level {expected_prefix!r}; "
            f"got {sorted(prefixes)}"
        )
    prefix = expected_prefix + "/"

    chunk_bytes = read_member(zf, prefix + "chunk.json")
    hashes_bytes = read_member(zf, prefix + "manifest_hashes.json")
    train_bytes = read_member(zf, prefix + "manifests/panda_train.csv")
    val_bytes = read_member(zf, prefix + "manifests/panda_validation.csv")

    meta = json.loads(chunk_bytes)
    manifest_hashes = json.loads(hashes_bytes)
    if int(meta["chunk_index"]) != index:
        zf.close()
        raise ValueError(
            f"Chunk index mismatch inside {path.name}: {meta['chunk_index']} != {index}"
        )

    for name, payload in (
        ("panda_train.csv", train_bytes),
        ("panda_validation.csv", val_bytes),
    ):
        expected = manifest_hashes.get(name)
        actual = sha256_bytes(payload)
        if expected != actual:
            zf.close()
            raise ValueError(
                f"Manifest checksum mismatch inside {path.name}: {name}"
            )

    train_fields, train_rows = read_csv_bytes(train_bytes)
    val_fields, val_rows = read_csv_bytes(val_bytes)
    if train_fields != val_fields:
        zf.close()
        raise ValueError(f"Train/validation manifest schemas differ in {path.name}")

    return {
        "index": index,
        "path": path,
        "archive_sha256": archive_sha,
        "zip": zf,
        "prefix": prefix,
        "meta": meta,
        "chunk_json_sha256": sha256_bytes(chunk_bytes),
        "fields": train_fields,
        "train": train_rows,
        "validation": val_rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunks-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="allow a preliminary incomplete chunk set; marks sample_only=true",
    )
    args = p.parse_args()

    archives = discover_archives(args.chunks_dir)
    chunks = []
    try:
        chunks = [open_chunk(*item) for item in archives]

        indices = [c["index"] for c in chunks]
        expected_prefix = list(range(indices[-1] + 1))
        contiguous = indices == expected_prefix
        final_indices = [
            c["index"] for c in chunks if c["meta"].get("is_last_chunk")
        ]
        has_final = bool(final_indices)

        if len(final_indices) > 1:
            raise ValueError(f"Multiple chunks claim is_last_chunk=true: {final_indices}")

        if not args.allow_partial:
            if not contiguous:
                raise ValueError(
                    f"Full preparation requires chunk indices 0..{indices[-1]}; got {indices}"
                )
            if not has_final:
                raise ValueError(
                    "No uploaded shard is marked is_last_chunk=true; use "
                    "--allow-partial only for a deliberate preliminary run"
                )
            if final_indices[0] != indices[-1]:
                raise ValueError("Chunks exist after the shard marked as final")

        first = chunks[0]["meta"]
        compatibility_keys = [
            "global_eligible",
            "target_fov_um",
            "output_pixels",
            "reference_sha256",
            "inventory_settings_sha256",
            "selection_order",
            "training_seed_definition",
        ]
        for chunk in chunks[1:]:
            for key in compatibility_keys:
                if chunk["meta"].get(key) != first.get(key):
                    raise ValueError(
                        f"Chunk {chunk['index']} incompatible on {key}: "
                        f"{chunk['meta'].get(key)!r} != {first.get(key)!r}"
                    )

        intervals = sorted(
            (
                int(c["meta"]["selection_start"]),
                int(c["meta"]["selection_stop"]),
                c["index"],
            )
            for c in chunks
        )
        expected_start = 0
        for start, stop, idx in intervals:
            if stop < start:
                raise ValueError(f"Invalid interval for chunk {idx}: {start}:{stop}")
            if not args.allow_partial and start != expected_start:
                raise ValueError(
                    f"Gap/overlap before chunk {idx}: expected start "
                    f"{expected_start}, got {start}"
                )
            if args.allow_partial and start < expected_start:
                raise ValueError(f"Overlapping chunk interval at chunk {idx}")
            expected_start = max(expected_start, stop)

        output = args.output.resolve()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output must be new/empty: {output}")
        manifests = output / "manifests"
        cache = output / "cache"
        manifests.mkdir(parents=True, exist_ok=True)
        cache.mkdir(parents=True, exist_ok=True)

        all_rows = {"train": [], "validation": []}
        seen_sample_ids = set()
        seen_tile_ids = set()
        seen_cache_names: dict[str, str] = {}
        fields = chunks[0]["fields"]

        for chunk in chunks:
            if chunk["fields"] != fields:
                raise ValueError(f"Manifest schema differs in chunk {chunk['index']}")
            zf = chunk["zip"]
            prefix = chunk["prefix"]
            for split in ("train", "validation"):
                for row in chunk[split]:
                    if row["dataset"] != "panda" or row["split"] != split:
                        raise ValueError(
                            f"Invalid dataset/split in chunk {chunk['index']}: {row['sample_id']}"
                        )
                    label = int(row["label"])
                    if label not in (0, 1, 2) or row["class_name"] != CLASSES[label]:
                        raise ValueError(f"Invalid class mapping: {row['sample_id']}")
                    if row["sample_id"] in seen_sample_ids:
                        raise ValueError(f"Duplicate sample_id: {row['sample_id']}")
                    if row["tile_id"] in seen_tile_ids:
                        raise ValueError(f"Duplicate tile_id: {row['tile_id']}")
                    seen_sample_ids.add(row["sample_id"])
                    seen_tile_ids.add(row["tile_id"])

                    name = Path(row["cache_file"]).name
                    if name != row["cache_file"]:
                        raise ValueError(f"Unsafe cache filename: {row['cache_file']}")
                    expected_sha = row["cache_sha256"]
                    if re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
                        raise ValueError(f"Invalid cache SHA-256: {row['sample_id']}")
                    prior = seen_cache_names.get(name)
                    if prior is not None and prior != expected_sha:
                        raise ValueError(f"Cache basename collision: {name}")
                    seen_cache_names[name] = expected_sha

                    stream_cache_member(
                        zf,
                        prefix + "cache/" + name,
                        cache / name,
                        expected_sha,
                    )
                    all_rows[split].append(row)

        for split in ("train", "validation"):
            all_rows[split].sort(
                key=lambda row: (
                    row["image_id"],
                    int(row["y"]),
                    int(row["x"]),
                    row["tile_id"],
                )
            )
            write_csv(manifests / f"panda_{split}.csv", all_rows[split], fields)

        train_slides = {row["source_wsi"] for row in all_rows["train"]}
        validation_slides = {row["source_wsi"] for row in all_rows["validation"]}
        if train_slides & validation_slides:
            raise ValueError("Slide leakage across train/validation in uploaded chunks")

        counts = {}
        for split in ("train", "validation"):
            counter = Counter(int(row["label"]) for row in all_rows[split])
            counts[split] = {CLASSES[k]: int(counter.get(k, 0)) for k in range(3)}

        train_classes = {int(row["label"]) for row in all_rows["train"]}
        training_ready = bool(
            all_rows["train"]
            and all_rows["validation"]
            and train_classes == {0, 1, 2}
        )
        if not training_ready:
            raise ValueError(
                "Uploaded chunk set is not train-ready: need nonempty validation "
                "and GP3/GP4/GP5 in training"
            )

        full_cohort = bool(
            contiguous
            and has_final
            and intervals[0][0] == 0
            and intervals[-1][1] == int(first["global_eligible"])
            and final_indices[0] == indices[-1]
        )
        if not args.allow_partial and not full_cohort:
            raise ValueError("Uploaded chunks do not cover the complete eligible cohort")

        eligibility = {
            "dataset": "PANDA",
            "provider": "radboud",
            "chunks": indices,
            "chunk_json_sha256": {
                str(c["index"]): c["chunk_json_sha256"] for c in chunks
            },
            "chunk_archive_sha256": {
                str(c["index"]): c["archive_sha256"] for c in chunks
            },
            "global_eligible": first["global_eligible"],
            "training_seed_definition": first["training_seed_definition"],
            "patient_independence": first["patient_independence"],
            "target_fov_um": first["target_fov_um"],
            "output_pixels": first["output_pixels"],
        }
        eligibility_path = output / "panda_source_eligibility.json"
        atomic_json(eligibility_path, eligibility)

        manifest_hashes = {
            f"panda_{split}.csv": sha256_file(manifests / f"panda_{split}.csv")
            for split in ("train", "validation")
        }
        archive_records = [
            {
                "chunk_index": c["index"],
                "filename": c["path"].name,
                "sha256": c["archive_sha256"],
            }
            for c in chunks
        ]

        prepared = {
            "complete": True,
            "training_ready": True,
            "sample_only": not full_cohort,
            "preliminary": not full_cohort,
            "datasets": ["panda"],
            "class_order": CLASSES,
            "manifests": manifest_hashes,
            "counts": counts,
            "seed_definition": first["training_seed_definition"],
            "region_grade_sha256": sha256_file(eligibility_path),
            "reference_sha256": first["reference_sha256"],
            "preprocessing": first["preprocessing"],
            "prenormalized_sources_accepted": False,
            "source_dataset": "PANDA/Radboud",
            "evaluation_split": "validation",
            "target_fov_um": first["target_fov_um"],
            "output_pixels": first["output_pixels"],
            "patient_independence": first["patient_independence"],
            "chunks": indices,
            "chunk_archives": archive_records,
            "global_eligible": first["global_eligible"],
            "merged_retained": sum(len(v) for v in all_rows.values()),
            "full_chunk_coverage": full_cohort,
            "preparation_mode": "direct_verified_chunk_zips",
            "source_chunks_dir": str(Path(args.chunks_dir).resolve()),
        }
        atomic_json(output / "prepared.json", prepared)

        print(json.dumps(prepared, indent=2))
        print("PBS-ready directory:", output)
        if prepared["sample_only"]:
            print(
                "WARNING: preliminary partial PANDA cohort. Production-like "
                "training requires an explicit allow-partial flag and must be "
                "reported as preliminary/bounded, not full-cohort training."
            )
        else:
            print("Full contiguous PANDA chunk cohort assembled.")

    finally:
        for chunk in chunks:
            try:
                chunk["zip"].close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
