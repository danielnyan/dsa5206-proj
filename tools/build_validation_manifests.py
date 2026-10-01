"""Stage 2B: read-only image inventory; UTF-8/LF deterministic CSV artifacts.

Uses Pillow headers/verify for native metadata, SHA-256 of every file's bytes,
and bounded parallel reads. Full pixel decoding was performed in Stage 2A.
No TensorFlow, resampling, normalization, splitting, image writes or downloads.
Relative paths are class-prefixed NFC POSIX paths relative to each archive root.
Every CSV has its own zero-based row_index; sample_id is the cross-file identity.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import unicodedata

from PIL import Image

FIELDS = ["row_index", "sample_id", "cohort", "class_name", "ground_truth_code",
          "relative_path", "file_sha256", "file_size_bytes", "width", "height", "mode"]
CLASSES = {"benign": 0, "tumour": 1}
COUNTS = {"VAL1": {"benign": 93009, "tumour": 52810}, "VAL2": {"benign": 24471, "tumour": 9632}}
ARCHIVES = {(cohort, label): f"val_dataset_{number}_{suffix}" for number, cohort in [(1, "VAL1"), (2, "VAL2")]
            for label, suffix in [("benign", "norm"), ("tumour", "tu")]}
LOG = logging.getLogger("validation_manifests")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalized_path(value: str) -> str:
    value = unicodedata.normalize("NFC", value.replace("\\", "/"))
    if value.startswith("/") or any(part in ("", ".", "..") or ":" in part for part in value.split("/")):
        raise ValueError(f"Unsafe relative path: {value}")
    return value


def inventory(root: Path) -> dict[str, Path]:
    root = root.resolve(strict=True)
    found, folded = {}, set()
    def fail(error):
        raise error
    for parent, dirs, files in os.walk(root, followlinks=False, onerror=fail):
        for name in dirs + files:
            path = Path(parent) / name
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                raise ValueError(f"Links/junctions are unsupported: {path}")
        for name in files:
            path = Path(parent) / name
            relative = normalized_path(path.relative_to(root).as_posix())
            if relative.casefold() in folded:
                raise ValueError(f"Case/Unicode path collision: {relative}")
            folded.add(relative.casefold())
            found[relative] = path
    return dict(sorted(found.items()))


def inspect_image(task: tuple[str, str, str, Path]) -> tuple[dict, tuple]:
    cohort, label, relative, path = task
    before = path.stat()
    payload = path.read_bytes()
    if not payload:
        raise ValueError(f"Empty image: {path}")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            width, height, mode = image.width, image.height, image.mode
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Multi-frame image")
            image.verify()
    except Exception as error:
        raise ValueError(f"Unreadable/non-image file {path}: {error}") from error
    after = path.stat()
    signature = lambda stat: (stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino)
    if signature(before) != signature(after) or len(payload) != after.st_size:
        raise ValueError(f"Image changed while reading: {path}")
    logical = label + "/" + relative
    return dict(sample_id=f"{cohort}:{logical}", cohort=cohort, class_name=label,
                ground_truth_code=CLASSES[label], relative_path=logical, file_sha256=sha256(payload),
                file_size_bytes=len(payload), width=width, height=height, mode=mode), signature(after)


def ordered_rows(rows: list[dict]) -> list[dict]:
    return [dict(row, row_index=index) for index, row in enumerate(sorted(
        rows, key=lambda row: (row["cohort"], row["class_name"], normalized_path(row["relative_path"]))))]


def validate_rows(rows: list[dict], cohort: str, expected: dict[str, int]) -> None:
    if Counter(row["class_name"] for row in rows) != Counter(expected):
        raise ValueError(f"Class counts do not match expected counts for {cohort}")
    if rows != ordered_rows(rows):
        raise ValueError("Rows/indices are not in canonical order")
    ids, indices, paths = set(), set(), set()
    for row in rows:
        label, relative = row["class_name"], row["relative_path"]
        if label not in CLASSES or row["cohort"] != cohort or row["ground_truth_code"] != CLASSES[label]:
            raise ValueError("Invalid cohort/class/ground-truth mapping")
        if relative != normalized_path(relative) or not relative.startswith(label + "/"):
            raise ValueError("Invalid class-relative path")
        if row["sample_id"] != cohort + ":" + relative:
            raise ValueError("Invalid stable sample_id")
        key = (label, relative.casefold())
        if row["sample_id"] in ids or row["row_index"] in indices or key in paths:
            raise ValueError("Duplicate identity/index/normalized path")
        ids.add(row["sample_id"]); indices.add(row["row_index"]); paths.add(key)
        if not re.fullmatch(r"[0-9a-f]{64}", row.get("file_sha256", "")):
            raise ValueError("Missing/invalid SHA-256")
        if any(type(row.get(k)) is not int or row[k] <= 0 for k in ("file_size_bytes", "width", "height")) or not row.get("mode"):
            raise ValueError("Missing/invalid native metadata")


def csv_bytes(rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def build_cohort(cohort: str, roots: dict[str, Path], expected: dict[str, int], workers: int = 8) -> list[dict]:
    all_rows = []
    for label in sorted(CLASSES):
        paths = inventory(roots[label])
        if len(paths) != expected[label]:
            raise ValueError(f"Unexpected file count for {cohort}/{label}: {len(paths)} != {expected[label]}")
        tasks = [(cohort, label, relative, path) for relative, path in paths.items()]
        signatures = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for offset in range(0, len(tasks), 1000):
                for task, (row, signature) in zip(tasks[offset:offset+1000], pool.map(inspect_image, tasks[offset:offset+1000])):
                    all_rows.append(row)
                    signatures[task[2]] = signature
                if (offset // 1000) % 5 == 0 or offset+1000 >= len(tasks):
                    LOG.info("%s/%s: %d/%d hashed", cohort, label, min(offset+1000, len(tasks)), len(tasks))
        current = inventory(roots[label])
        if current != paths:
            raise ValueError("Dataset inventory changed during generation")
        for relative, path in current.items():
            s = path.stat()
            if signatures[relative] != (s.st_size, s.st_mtime_ns, s.st_dev, s.st_ino):
                raise ValueError(f"Dataset changed during generation: {path}")
    rows = ordered_rows(all_rows)
    validate_rows(rows, cohort, expected)
    return rows


def distribution(rows: list[dict]) -> dict:
    return {"dimensions": dict(sorted(Counter(f"{r['width']}x{r['height']}" for r in rows).items())),
            "modes": dict(sorted(Counter(r["mode"] for r in rows).items()))}


def evaluator_compatibility(repo: Path) -> dict:
    path = repo / "modern_pca/evaluate_paper.py"
    reference = "1bf996d"
    result = {"current_evaluator_exists": path.exists(), "historical_reference": reference,
              "status": "UNVERIFIED_CURRENT_CHECKOUT",
              "organization": "Historical evaluator accepts archive roots via --benign-dir and --tumour-dir, recursively preserving norm/tu subdirectories.",
              "csv_import": "Historical evaluator constructs its own manifest; it has no direct input-manifest option. A future CSV adapter would map class_name to source_group and ground_truth, width/height/mode to source_width/source_height/source_mode.",
              "hash_warning": "Stage 2B CSV fingerprints differ from evaluator internal CSV fingerprints because schemas differ. Do not pass these hashes as --expected-manifest-sha256.",
              "action": "Restore/review the evaluator in the intended branch before evaluation; no evaluator changes made."}
    try:
        historical = subprocess.check_output(["git", "show", reference+":modern_pca/evaluate_paper.py"], cwd=repo)
        result["historical_source_sha256"] = sha256(historical)
        if path.exists():
            result["current_source_sha256"] = sha256(path.read_bytes())
            result["current_matches_historical_reference"] = path.read_bytes().replace(b"\r\n", b"\n") == historical.replace(b"\r\n", b"\n")
    except subprocess.CalledProcessError:
        result["historical_source_available"] = False
    return result


def generate(dataset: Path, output: Path, report_path: Path, workers: int = 8) -> dict:
    if workers < 1:
        raise ValueError("workers must be positive")
    dataset, output = dataset.resolve(strict=True), output.resolve()
    if output.exists():
        raise ValueError("Output directory exists; refusing overwrite")
    if dataset == output or dataset in output.parents or output in dataset.parents:
        raise ValueError("Dataset and output must not overlap")
    report_payload = report_path.read_bytes()
    stage2a = json.loads(report_payload)
    if stage2a.get("status") != "verification_complete" or not stage2a.get("all_archive_md5_passed"):
        raise ValueError("Stage 2A verification must be complete with all MD5 checks passed")
    artifacts, cohorts, hashes = {}, {}, set()
    for cohort in COUNTS:
        roots = {label: dataset / "extracted" / ARCHIVES[cohort, label] for label in CLASSES}
        rows = build_cohort(cohort, roots, COUNTS[cohort], workers)
        for row in rows:
            if row["file_sha256"] in hashes:
                raise ValueError("Duplicate image content found; differs from Stage 2A findings")
            hashes.add(row["file_sha256"])
        classes = {}
        for label in CLASSES:
            subset = ordered_rows([r for r in rows if r["class_name"] == label])
            validate_rows(subset, cohort, {label: COUNTS[cohort][label]})
            classes[label] = dict(count=len(subset), source_root=str(roots[label]), **distribution(subset))
            prior = [a for a in stage2a["archives"] if a["cohort"] == cohort and a["class_name"] == label]
            if len(prior) != 1 or prior[0]["image_files"] != len(subset) or any(prior[0][k] != classes[label][k] for k in ("dimensions", "modes")):
                raise ValueError("Stage 2A counts/native metadata do not match")
            artifacts[f"{cohort}_{label}.csv"] = csv_bytes(subset)
        name = f"{cohort}_manifest.csv"
        artifacts[name] = csv_bytes(rows)
        cohorts[cohort] = dict(total=len(rows), classes=classes, manifest_sha256=sha256(artifacts[name]), **distribution(rows))
    dimension_variants = sum(info["count"] - max(info["dimensions"].values())
                             for cohort in cohorts.values() for info in cohort["classes"].values())
    summary = dict(schema_version=1, status="PASS", dataset_source="Zenodo 3825933",
                   generator_source_sha256=sha256(Path(__file__).read_bytes()),
                   native_dimension_variants=dimension_variants,
                   generated_at_utc=datetime.now(timezone.utc).isoformat(), source_images_modified=False,
                   source_image_note="Only read/stat operations were used on source images; no resizing, renaming, moving or splitting.",
                   stage_2a_reports={"json": str(report_path.resolve()), "json_sha256": sha256(report_payload),
                                     "markdown": str(report_path.with_suffix('.md').resolve())},
                   cohorts=cohorts, total_images=sum(c["total"] for c in cohorts.values()),
                   fields=FIELDS, csv_encoding="UTF-8 without BOM; LF newlines",
                   sort_order=["cohort", "class_name", "NFC POSIX relative_path (case-sensitive lexical)"],
                   row_index_policy="Zero-based contiguous indices independently assigned after sorting each CSV; join on sample_id.",
                   path_policy="relative_path = class_name/archive-relative-path; resolve by stripping class_name/ and joining the corresponding source_root; sample_id = cohort:relative_path.",
                   validation={"counts": "PASS", "unique_id_index_path": "PASS", "sha256_every_image": "PASS",
                               "native_metadata": "PASS", "all_files_exactly_once_per_cohort": "PASS",
                               "stage_2a_distribution_comparison": "PASS", "no_duplicate_contents": "PASS"},
                   evaluator_compatibility=evaluator_compatibility(Path(__file__).resolve().parents[1]),
                   artifacts={name: dict(sha256=sha256(payload), size_bytes=len(payload)) for name, payload in artifacts.items()})
    # Validate serialization before publishing any outputs. Summary JSON is last.
    for name, payload in artifacts.items():
        reread = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
        if not reread or list(reread[0]) != FIELDS or any(int(row["row_index"]) != i for i, row in enumerate(reread)):
            raise ValueError(f"CSV round-trip failed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in artifacts.items():
        path = output / name
        path.write_bytes(payload)
        if sha256(path.read_bytes()) != summary["artifacts"][name]["sha256"]:
            raise ValueError(f"Written artifact hash mismatch: {name}")
    lines = ["# Stage 2B manifest summary", "", "Source: Zenodo 3825933. Validation: **PASS**.", "",
             "Source images were not modified. No resizing, resampling, renaming, moving, random splitting or cohort merging occurred.", "",
             "| Cohort | Benign | Tumour | Total | Cohort CSV SHA-256 |", "|---|---:|---:|---:|---|"]
    for cohort, value in cohorts.items():
        lines.append(f"| {cohort} | {value['classes']['benign']['count']} | {value['classes']['tumour']['count']} | {value['total']} | {value['manifest_sha256']} |")
    lines += ["", "Generation UTC: " + summary["generated_at_utc"], "", "Stage 2A references: " + str(report_path) + " and " + str(report_path.with_suffix('.md')),
              "", summary["path_policy"], "", summary["row_index_policy"], "",
              f"All {dimension_variants:,} native dimension variants are retained. Native modes are listed below."]
    for cohort, value in cohorts.items():
        lines += ["", "## " + cohort, "", "Dimensions (width x height): " + json.dumps(value["dimensions"]), "Modes: " + json.dumps(value["modes"])]
        for label, info in value["classes"].items():
            lines += ["", f"{label} source root: {info['source_root']}", "Dimensions: " + json.dumps(info["dimensions"]), "Modes: " + json.dumps(info["modes"])]
    compatibility = summary["evaluator_compatibility"]
    lines += ["", "## Evaluator compatibility", "", "Current evaluate_paper.py present: " + str(compatibility["current_evaluator_exists"]),
              "Historical source reviewed: " + compatibility["historical_reference"], "", compatibility["organization"],
              "", compatibility["csv_import"], "", compatibility["hash_warning"], "", compatibility["action"],
              "", "Stage 2C data preparation can proceed. Evaluator execution remains pending restoration/review of the missing evaluator and its other prerequisites.", ""]
    (output / "manifest_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "manifest_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("manifests"))
    parser.add_argument("--stage2a-report", type=Path, default=Path("reports/dataset_verification.json"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    result = generate(args.dataset_root, args.output, args.stage2a_report, args.workers)
    for cohort, value in result["cohorts"].items():
        LOG.info("%s: %d images; SHA-256=%s", cohort, value["total"], value["manifest_sha256"])


if __name__ == "__main__":
    main()
