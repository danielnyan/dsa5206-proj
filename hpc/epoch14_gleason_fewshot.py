#!/usr/bin/env python3
"""Frozen epoch-14 binary NASNet -> PANDA GP3/GP4/GP5 representation transfer.

Nine conditions: F44/F22/F11 GAP x {zero_fit_1nn, prototype_5, prototype_50}.
The binary model is never trained or updated. 'Zero-fit' uses labelled PANDA
training embeddings as a retrieval bank, not a genuinely label-free predictor.

Run under the project Python environment on a GPU compute node. --preflight
runs without TensorFlow/GPU and verifies the manifest/sampling contract.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

STAGES = {"F44": ("normal_concat_5", 1008),
          "F22": ("normal_concat_12", 2016),
          "F11": ("backbone_output_post_relu", 4032)}
CLASS_NAMES = ("GP3", "GP4", "GP5")
EXPECTED_CHECKPOINT_SHA256 = "433dd48aae800c659d1cba092af2859499bb804d2228ff978a6e9c9602b2adde"
PROTOCOL_VERSION = "epoch14-panda-gp-transfer-v1"


def sha(path: Path, block_size=8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False, sort_keys=True) + "\n")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict], columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def git_info():
    try:
        return {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip(),
            "status": subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "status": "unavailable"}


def eligible_pure(row):
    return (row.get("region_role") == "pure"
            and str(row.get("region_primary")) == str(row.get("region_secondary"))
            and str(row.get("region_primary")) == str(int(row["label"]) + 3))


def read_population(prepared_dir: Path):
    # Reuse the canonical checksum-verified manifest reader and leakage guard.
    from modern_pca.gleason import read_rows, assert_training_isolated

    pfile = prepared_dir / "prepared.json"
    prepared = json.loads(pfile.read_text())
    if prepared.get("datasets") != ["panda"] or not prepared.get("complete") or not prepared.get("training_ready"):
        raise ValueError("Need a complete, training-ready PANDA preparation")
    if prepared.get("sample_only") or not prepared.get("full_chunk_coverage"):
        raise ValueError("This benchmark requires full PANDA chunks (sample_only=False; full_chunk_coverage=True)")
    manifest_root = prepared_dir / "manifests"
    train_all = read_rows(manifest_root, "panda", "train")
    val_all = read_rows(manifest_root, "panda", "validation")
    assert_training_isolated(train_all, val_all)

    excluded = []
    populations = {}
    for split, all_rows in (("train", train_all), ("validation", val_all)):
        pure = []
        for row in all_rows:
            if eligible_pure(row):
                pure.append(row)
            else:
                excluded.append({"split": split, "sample_id": row["sample_id"], "reason": "non_pure_region"})

        # Within each split, remove exact-byte duplicates and all conflicting-label
        # image contents. Keeping one identical PNG avoids pseudo-replication.
        by_content = defaultdict(list)
        for row in pure:
            if not row.get("file_sha256"):
                raise ValueError("Missing file_sha256: " + row["sample_id"])
            by_content[row["file_sha256"]].append(row)
        cleaned = []
        for digest, group in sorted(by_content.items()):
            labels = {r["label"] for r in group}
            if len(labels) > 1:
                for r in group:
                    excluded.append({"split": split, "sample_id": r["sample_id"], "reason": "same_image_conflicting_gp_labels"})
                continue
            # Representative chosen canonically, never based on model output.
            group.sort(key=lambda r: r["sample_id"])
            cleaned.append(group[0])
            for r in group[1:]:
                excluded.append({"split": split, "sample_id": r["sample_id"], "reason": "same_image_duplicate_within_split"})
        cleaned.sort(key=lambda r: r["sample_id"])
        populations[split] = cleaned

    assert_training_isolated(populations["train"], populations["validation"])
    for split, rows in populations.items():
        counter = Counter(r["label"] for r in rows)
        if any(counter[k] == 0 for k in range(3)):
            raise ValueError(f"{split}: no eligible samples in one or more classes: {counter}")
    return prepared, populations, excluded


def slide_diverse_support(train: list[dict], seed: int, max_shots: int = 50):
    """Nested 5 in 50 supports; maximize distinct WSIs across all GP classes.

    First prefer a WSI not used in any class, then a WSI not yet used in
    this class, then a repeated WSI. Never select the same image content twice.
    Selection is predetermined from manifest identifiers, never embeddings.
    """
    by_class = defaultdict(list)
    for row in train:
        by_class[row["label"]].append(row)
    for c in range(3):
        if len(by_class[c]) < max_shots:
            raise ValueError(f"Need {max_shots} distinct-content GP{c+3} images, found {len(by_class[c])}")

    rng = random.Random(seed)
    pools = {}
    for c in range(3):
        by_slide = defaultdict(list)
        for r in by_class[c]:
            by_slide[r["source_wsi"]].append(r)
        slides = sorted(by_slide)
        rng.shuffle(slides)
        for slide in slides:
            group = sorted(by_slide[slide], key=lambda r: r["sample_id"])
            rng.shuffle(group)
            by_slide[slide] = group
        # Round robin: every slide's first image before second image from a slide.
        ordered = []
        for position in range(max(map(len, by_slide.values()))):
            for slide in slides:
                if len(by_slide[slide]) > position:
                    ordered.append(by_slide[slide][position])
        pools[c] = ordered

    global_slides, global_patients = set(), set()
    global_hashes = set()
    class_slides = {c: set() for c in range(3)}
    class_patients = {c: set() for c in range(3)}
    chosen = {c: [] for c in range(3)}
    audit = []
    for rank in range(1, max_shots + 1):
        for c in range(3):
            eligible = [r for r in pools[c] if r["file_sha256"] not in global_hashes]
            if not eligible:
                raise ValueError(f"Insufficient nonduplicated GP{c+3} support samples")
            def distinct(r, wsis, patients):
                return (r["source_wsi"] not in wsis and
                        (not r.get("patient_id") or r["patient_id"] not in patients))
            candidates = [r for r in eligible if distinct(r, global_slides, global_patients)]
            tier = "new_wsi_and_known_patient_global"
            if not candidates:
                candidates = [r for r in eligible if distinct(r, class_slides[c], class_patients[c])]
                tier = "new_wsi_and_known_patient_within_class"
            if not candidates:
                candidates = eligible
                tier = "reused_wsi_or_known_patient"
            row = candidates[0]
            chosen[c].append(row)
            global_hashes.add(row["file_sha256"])
            global_slides.add(row["source_wsi"])
            class_slides[c].add(row["source_wsi"])
            if row.get("patient_id"):
                global_patients.add(row["patient_id"])
                class_patients[c].add(row["patient_id"])
            audit.append({
                "seed": seed, "shot_rank": rank, "class_name": CLASS_NAMES[c],
                "label": c, "sample_id": row["sample_id"],
                "image_id": row.get("image_id", ""), "source_wsi": row["source_wsi"],
                "patient_id": row.get("patient_id", ""),
                "x": row.get("x", ""), "y": row.get("y", ""),
                "tile_id": row.get("tile_id", ""),
                "file_sha256": row["file_sha256"], "cache_sha256": row["cache_sha256"],
                "cache_file": row["cache_file"], "selection_tier": tier,
                "included_in_5shot": rank <= 5,
                "included_in_50shot": True,
            })
    return chosen, audit


def make_supports(train, seeds: list[int]):
    selected = {}
    audit = []
    summaries = []
    index = {r["sample_id"]: i for i, r in enumerate(train)}
    for seed in seeds:
        chosen, records = slide_diverse_support(train, seed=seed)
        audit.extend(records)
        for shots in (5, 50):
            rows = [r for c in range(3) for r in chosen[c][:shots]]
            idx = np.array([index[r["sample_id"]] for r in rows], dtype=np.int64)
            labels = np.array([r["label"] for r in rows], dtype=np.int32)
            assert len(rows) == 3 * shots and all(sum(labels == c) == shots for c in range(3))
            selected[(seed, shots)] = (idx, labels)
            for c in range(3):
                part = chosen[c][:shots]
                summaries.append({
                    "seed": seed, "shots": shots, "class_name": CLASS_NAMES[c],
                    "n_images": len(part),
                    "n_unique_wsi": len({r["source_wsi"] for r in part}),
                    "n_known_patients": len({r["patient_id"] for r in part if r.get("patient_id")}),
                    "n_unique_contents": len({r["file_sha256"] for r in part}),
                    "n_reused_wsi_or_known_patient": sum(x["selection_tier"] == "reused_wsi_or_known_patient" for x in records
                                        if x["label"] == c and x["shot_rank"] <= shots),
                })
    return selected, audit, summaries


def unit_normalize(x: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise ValueError("Nonfinite or malformed embeddings")
    norm = np.linalg.norm(x.astype(np.float64), axis=1).astype(np.float32)
    if np.any(norm < 1e-12):
        raise ValueError("Zero-norm embedding; cosine representation undefined")
    return x / norm[:, None]


def compute_metrics(truth, pred, labels=(0, 1, 2)):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, confusion_matrix, precision_recall_fscore_support)
    truth = np.asarray(truth, dtype=int)
    pred = np.asarray(pred, dtype=int)
    pr, rc, f1, support = precision_recall_fscore_support(truth, pred, labels=list(labels), zero_division=0)
    return {
        "n_evaluation": len(truth),
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, labels=list(labels), average="macro", zero_division=0)),
        "class_precision": [float(x) for x in pr],
        "class_recall": [float(x) for x in rc],
        "class_f1": [float(x) for x in f1],
        "class_support": [int(x) for x in support],
        "confusion_matrix": confusion_matrix(truth, pred, labels=list(labels)).tolist(),
    }


def prototype_predict(train_embeddings: np.ndarray, val_embeddings: np.ndarray,
                      support_indices: np.ndarray, support_labels: np.ndarray):
    """Normalize each image, mean per class, renormalize each mean, cosine decision."""
    support = unit_normalize(np.asarray(train_embeddings[support_indices]))
    prototype = []
    for c in range(3):
        mean = support[support_labels == c].mean(axis=0)
        prototype.append(mean)
    proto = unit_normalize(np.array(prototype, dtype=np.float32))
    # query features already unit normalized by caller; block multiplication
    scores = val_embeddings @ proto.T
    pred = scores.argmax(axis=1).astype(np.int32)
    return pred, scores, proto


def zero_fit_nn(train_embeddings: np.ndarray, query_embeddings: np.ndarray,
                train_rows: list[dict], val_rows: list[dict],
                query_block=128, bank_block=2048, device="gpu"):
    """Exact cosine 1-NN in a WSI/patient-disjoint PANDA training bank.

    Chunked CPU dot-products keep RAM bounded. Ties choose the earliest
    canonical training index because comparisons use strict '>'.
    """
    nt, nq = len(train_rows), len(val_rows)
    if train_embeddings.shape[0] != nt or query_embeddings.shape[0] != nq:
        raise ValueError("Embedding length and manifest length disagree")
    nn_indices = np.empty(nq, dtype=np.int32)
    nn_cosines = np.empty(nq, dtype=np.float32)
    # GPU approach stores only the normalized bank and small query batches,
    # never a full query x bank similarity matrix. CPU chunked fallback works
    # without additional packages. 1NN reference features are NOT fitted.
    tf = None
    gpu_bank = None
    if device == "gpu":
        import tensorflow as tf
        if not tf.config.list_physical_devices("GPU"):
            raise RuntimeError("--nn-device gpu requested but no GPU visible")
        with tf.device("/GPU:0"):
            gpu_bank = tf.constant(np.asarray(train_embeddings, dtype=np.float32))
    for qstart in range(0, nq, query_block):
        qstop = min(nq, qstart + query_block)
        Q = np.asarray(query_embeddings[qstart:qstop], dtype=np.float32)
        best_val = np.full(len(Q), -np.inf, dtype=np.float32)
        best_idx = np.full(len(Q), -1, dtype=np.int32)
        if gpu_bank is not None:
            with tf.device("/GPU:0"):
                sims_tf = tf.linalg.matmul(tf.convert_to_tensor(Q), gpu_bank, transpose_b=True)
                idx_tf = tf.argmax(sims_tf, axis=1, output_type=tf.int32)
                val_tf = tf.reduce_max(sims_tf, axis=1)
            best_idx[:] = idx_tf.numpy()
            best_val[:] = val_tf.numpy()
        else:
            for bstart in range(0, nt, bank_block):
                bstop = min(nt, bstart + bank_block)
                B = np.asarray(train_embeddings[bstart:bstop], dtype=np.float32)
                sims = Q @ B.T
                local_idx = sims.argmax(axis=1)
                local_val = sims[np.arange(len(Q)), local_idx]
                better = local_val > best_val
                best_val[better] = local_val[better]
                best_idx[better] = bstart + local_idx[better]
        if np.any(best_idx < 0):
            raise RuntimeError("No 1NN reference found")
        # In this dataset all train/validation WSIs are disjoint, but validate
        # every returned match nonetheless.
        for i, j in enumerate(best_idx):
            q, ref = val_rows[qstart+i], train_rows[int(j)]
            if (q["source_wsi"] == ref["source_wsi"]
                    or q["file_sha256"] == ref["file_sha256"]
                    or (q.get("patient_id") and q["patient_id"] == ref.get("patient_id"))):
                raise ValueError("1NN leakage: held-out query shares slide, content, or patient")
        nn_indices[qstart:qstop] = best_idx
        nn_cosines[qstart:qstop] = best_val
        print(f"Zero-fit 1NN: {qstop}/{nq}", flush=True)
    return nn_indices, nn_cosines


def prepare_feature_cache(cache_dir: Path, train_rows, val_rows, run_config: dict, resume: bool):
    """Memory-mapped feature files with last-complete-batch progress."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "version": PROTOCOL_VERSION,
        "checkpoint_sha256": run_config["checkpoint_sha256"],
        "prepared_manifest_hashes": run_config["prepared_manifest_hashes"],
        "population_sample_id_sha256": run_config["population_sample_id_sha256"],
        "dimensions": {k: dim for k, (_, dim) in STAGES.items()},
    }
    metadata = cache_dir / "identity.json"
    if metadata.exists():
        if not resume:
            raise FileExistsError("Existing feature cache: rerun with --resume to reuse validated output")
        if json.loads(metadata.read_text()) != identity:
            raise ValueError("Feature-cache identity mismatch; choose a fresh output directory")
    else:
        if any(cache_dir.iterdir()):
            raise FileExistsError("Nonempty feature directory without identity.json")
        atomic_json(metadata, identity)

    arrays = {}
    progress = {}
    for split, rows in (("train", train_rows), ("validation", val_rows)):
        progress_file = cache_dir / f"{split}_progress.json"
        p = json.loads(progress_file.read_text()) if progress_file.exists() else {"completed": 0}
        completed = int(p["completed"])
        if not 0 <= completed <= len(rows):
            raise ValueError("Invalid feature progress")
        arrays[split] = {}
        for stage, (_, dim) in STAGES.items():
            path = cache_dir / f"{split}_{stage}.npy"
            if not path.exists():
                if completed:
                    raise RuntimeError("Missing feature file for nonzero progress")
                a = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(rows), dim))
                a.flush()
                del a
            a = np.load(path, mmap_mode="r+")
            if a.shape != (len(rows), dim) or a.dtype != np.float32:
                raise ValueError("Feature array shape/dtype mismatch: " + str(path))
            arrays[split][stage] = a
        progress[split] = completed
    return arrays, progress


def load_frozen_extractor(checkpoint: Path):
    import tensorflow as tf
    from modern_pca.models import build_classifier

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("This feature-extraction job requires an allocated GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.utils.set_random_seed(42)
    try:
        binary = tf.keras.models.load_model(checkpoint, compile=False)
        mode = "keras_load_model"
    except TypeError as err:
        if not ("input_axes" in str(err) or "output_axes" in str(err)):
            raise
        print("Keras initializer compatibility fallback: reconstructing topology and loading weights", flush=True)
        binary = build_classifier(2, weights=None, architecture="nasnetlarge", head="original", image_size=350)
        binary.load_weights(checkpoint)
        mode = "architecture_plus_weights"
    if binary.input_shape != (None, 350, 350, 3) or binary.output_shape != (None, 2):
        raise ValueError("Wrong epoch14 model input/output dimensions")
    backbones = [layer for layer in binary.layers if isinstance(layer, tf.keras.Model)
                 and any(inner.name == "normal_conv_1_1" for inner in layer.layers)]
    if len(backbones) != 1:
        raise RuntimeError(f"Expected one NASNetLarge backbone; found {len(backbones)}")
    backbone = backbones[0]
    binary.trainable = False
    backbone.trainable = False
    stage_base = tf.keras.Model(backbone.input, {
        "F44": backbone.get_layer("normal_concat_5").output,
        "F22": backbone.get_layer("normal_concat_12").output,
        "F11": backbone.output,
    })
    inp = tf.keras.Input(shape=(350, 350, 3), name="probe_input")
    maps = stage_base(inp, training=False)
    gap = {stage: tf.keras.layers.GlobalAveragePooling2D(name=f"{stage}_gap")(maps[stage])
           for stage in STAGES}
    extractor = tf.keras.Model(inp, gap, name="epoch14_frozen_three_stage_gap")
    extractor.trainable = False
    check_shape = {"F44": (44, 44, 1008), "F22": (22, 22, 2016), "F11": (11, 11, 4032)}
    for stage, expected in check_shape.items():
        actual = tuple(int(v) for v in stage_base.output[stage].shape[1:])
        if actual != expected:
            raise ValueError(f"Unexpected stage topology at {stage}: {actual} != {expected}")
    print(f"Loaded checkpoint using {mode}; frozen stages: {check_shape}", flush=True)
    return tf, extractor, mode


def extract_split(tf, extractor, rows, cache_root: Path, arrays, split: str,
                  offset: int, batch_size: int, features_root: Path):
    from modern_pca.gleason import cache_path
    paths = [str(cache_path(cache_root, r)) for r in rows[offset:]]
    ds = tf.data.Dataset.from_tensor_slices(paths)
    def decode(path):
        im = tf.io.decode_png(tf.io.read_file(path), channels=3)
        im = tf.ensure_shape(im, (350, 350, 3))
        return tf.cast(im, tf.float32) / 255.0
    ds = ds.map(decode, num_parallel_calls=2).batch(batch_size).prefetch(1)
    completed = offset
    t0 = time.monotonic()
    for batch in ds:
        gap = extractor(batch, training=False)
        count = int(batch.shape[0])
        for stage, target in arrays.items():
            values = gap[stage].numpy()
            if values.shape != (count, STAGES[stage][1]) or not np.all(np.isfinite(values)):
                raise RuntimeError(f"Malformed/nonfinite {stage} embedding at {split}:{completed}")
            target[completed:completed+count] = values
        completed += count
        if (completed - offset) % max(1, batch_size * 32) == 0 or completed == len(rows):
            for arr in arrays.values():
                arr.flush()
            atomic_json(features_root / f"{split}_progress.json", {"completed": completed, "total": len(rows), "updated_utc": utc()})
            print(f"Extract {split}: {completed}/{len(rows)} in {time.monotonic()-t0:.1f}s", flush=True)
    if completed != len(rows):
        raise RuntimeError("Incomplete extraction")
    return completed


def evaluate_all(run_dir: Path, features, populations, selected, seeds,
                 nn_query_block: int, nn_bank_block: int, nn_device: str):
    train = populations["train"]
    val = populations["validation"]
    truth = np.array([r["label"] for r in val], dtype=np.int32)
    predictions_dir = run_dir / "predictions"
    prototypes_dir = run_dir / "prototypes"
    predictions_dir.mkdir(exist_ok=True)
    prototypes_dir.mkdir(exist_ok=True)
    results = []
    for stage in STAGES:
        print("Evaluating stage", stage, flush=True)
        tvec = unit_normalize(features["train"][stage])
        qvec = unit_normalize(features["validation"][stage])
        nn_index, nn_sim = zero_fit_nn(tvec, qvec, train, val, nn_query_block, nn_bank_block, nn_device)
        preds = np.array([train[int(j)]["label"] for j in nn_index], dtype=int)
        metrics = compute_metrics(truth, preds)
        results.append({"stage": stage, "method": "zero_fit_1nn", "shots_per_class": 0,
                        "seed": "all", **metrics})
        rows_out = []
        for i, ref_i in enumerate(nn_index):
            q, ref = val[i], train[int(ref_i)]
            rows_out.append({
                "query_sample_id": q["sample_id"], "true_label": truth[i],
                "true_class": CLASS_NAMES[truth[i]],
                "predicted_label": int(preds[i]), "predicted_class": CLASS_NAMES[int(preds[i])],
                "query_wsi": q["source_wsi"], "query_file_sha256": q["file_sha256"],
                "reference_sample_id": ref["sample_id"], "reference_label": ref["label"],
                "reference_class": CLASS_NAMES[ref["label"]],
                "reference_wsi": ref["source_wsi"], "reference_file_sha256": ref["file_sha256"],
                "cosine_similarity": float(nn_sim[i]),
            })
        write_csv(predictions_dir / f"{stage}_zero_fit_1nn.csv", rows_out,
                  list(rows_out[0]))
        for shots in (5, 50):
            for seed in seeds:
                support_idx, support_labels = selected[(seed, shots)]
                pred, scores, proto = prototype_predict(tvec, qvec, support_idx, support_labels)
                np.save(prototypes_dir / f"{stage}_prototype_{shots}_seed{seed}.npy", proto)
                metrics = compute_metrics(truth, pred)
                results.append({"stage": stage, "method": f"prototype_{shots}",
                                "shots_per_class": shots, "seed": seed, **metrics})
                out = [{
                    "query_sample_id": r["sample_id"], "true_label": int(truth[i]),
                    "true_class": CLASS_NAMES[int(truth[i])],
                    "predicted_label": int(pred[i]), "predicted_class": CLASS_NAMES[int(pred[i])],
                    "query_wsi": r["source_wsi"], "query_file_sha256": r["file_sha256"],
                    "cosine_gp3": float(scores[i, 0]),
                    "cosine_gp4": float(scores[i, 1]),
                    "cosine_gp5": float(scores[i, 2]),
                } for i, r in enumerate(val)]
                write_csv(predictions_dir / f"{stage}_prototype_{shots}_seed{seed}.csv", out,
                          list(out[0]))
                print(f"{stage} {shots}-shot seed={seed}: macro-F1={metrics['macro_f1']:.4f}", flush=True)
    write_csv(run_dir / "metrics_by_seed.csv", [{
        "stage": x["stage"], "method": x["method"], "shots_per_class": x["shots_per_class"],
        "seed": x["seed"], "n_evaluation": x["n_evaluation"],
        "macro_f1": x["macro_f1"], "accuracy": x["accuracy"],
        "balanced_accuracy": x["balanced_accuracy"],
        **{f"recall_{CLASS_NAMES[i]}": x["class_recall"][i] for i in range(3)},
        **{f"f1_{CLASS_NAMES[i]}": x["class_f1"][i] for i in range(3)},
    } for x in results], ["stage", "method", "shots_per_class", "seed", "n_evaluation",
                        "macro_f1", "accuracy", "balanced_accuracy",
                        "recall_GP3", "recall_GP4", "recall_GP5", "f1_GP3", "f1_GP4", "f1_GP5"])
    atomic_json(run_dir / "metrics_detailed.json", results)
    summary = []
    for stage in STAGES:
        for method in ("zero_fit_1nn", "prototype_5", "prototype_50"):
            subset = [r for r in results if r["stage"] == stage and r["method"] == method]
            f1 = np.array([r["macro_f1"] for r in subset])
            ba = np.array([r["balanced_accuracy"] for r in subset])
            summary.append({"stage": stage, "method": method, "n_support_seeds": len(subset),
                            "macro_f1_mean": float(f1.mean()),
                            "macro_f1_sd": float(f1.std(ddof=1)) if len(f1) > 1 else None,
                            "balanced_accuracy_mean": float(ba.mean()),
                            "balanced_accuracy_sd": float(ba.std(ddof=1)) if len(ba) > 1 else None,
                            "n_evaluation": len(val)})
    write_csv(run_dir / "summary_9_conditions.csv", summary,
              ["stage", "method", "n_support_seeds", "n_evaluation", "macro_f1_mean",
               "macro_f1_sd", "balanced_accuracy_mean", "balanced_accuracy_sd"])
    atomic_json(run_dir / "summary_9_conditions.json", summary)
    return summary


def run(args):
    prepared_dir = args.prepared.resolve()
    prepared, populations, excluded = read_population(prepared_dir)
    train, val = populations["train"], populations["validation"]
    seeds = [args.seed + j for j in range(args.repeats)]
    selected, support_audit, support_summary = make_supports(train, seeds)
    counts = {split: {CLASS_NAMES[c]: sum(r["label"] == c for r in populations[split]) for c in range(3)}
              for split in ("train", "validation")}
    checks = {"population_counts": counts,
              "distinct_wsi": {split: len({r["source_wsi"] for r in populations[split]}) for split in populations},
              "excluded_count": len(excluded),
              "support_seeds": seeds,
              "support_5_and_50_nested": True,
              "patient_id_available": {split: any(bool(r.get("patient_id")) for r in populations[split]) for split in populations},
              "estimated_feature_gib": float(sum(len(rows) for rows in populations.values()) * sum(d for _,d in STAGES.values()) * 4 / 1024**3)}
    print("PRE-FLIGHT", json.dumps(checks, indent=2), flush=True)
    if args.preflight:
        return

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"epoch14 checkpoint not found: {checkpoint}")
    checkpoint_sha = sha(checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Wrong released binary checkpoint SHA-256: " + checkpoint_sha)

    if args.model_smoke:
        tf, extractor, mode = load_frozen_extractor(checkpoint)
        from modern_pca.gleason import cache_path
        row = train[0]
        image_path = cache_path(args.cache, row)
        if sha(image_path) != row["cache_sha256"]:
            raise ValueError("Smoke PNG SHA mismatch")
        image = tf.io.decode_png(tf.io.read_file(str(image_path)), channels=3)
        image = tf.cast(tf.ensure_shape(image, (350, 350, 3)), tf.float32) / 255.
        gap = extractor(image[None, ...], training=False)
        for stage, (_, dim) in STAGES.items():
            values = gap[stage].numpy()
            if values.shape != (1, dim) or not np.all(np.isfinite(values)):
                raise RuntimeError(f"Smoke result invalid at {stage}: {values.shape}")
        print("MODEL SMOKE PASS:", mode, "| three stage embeddings", flush=True)
        return

    run_dir = args.output.resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError("Refusing to overwrite nonempty output; use --resume or a new output")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol": PROTOCOL_VERSION, "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha, "prepared_dir": str(prepared_dir),
        "prepared_manifest_hashes": prepared["manifests"],
        "prepared_json_sha256": sha(prepared_dir / "prepared.json"),
        "population_sample_id_sha256": {split: hashlib.sha256("\n".join(r["sample_id"] for r in populations[split]).encode()).hexdigest()
                                        for split in ("train", "validation")},
        "seed": args.seed, "repeats": args.repeats, "batch_size": args.batch_size,
        "nn_device": args.nn_device, "nn_query_block": args.nn_query_block, "nn_bank_block": args.nn_bank_block,
        "support_shots": [5, 50], "stages": {k: {"layer": name, "dimension": dim} for k, (name, dim) in STAGES.items()},
        "classification": "frozen GAP embeddings; cosine 1NN / class-mean cosine prototypes",
        "label_use": "1NN bank labels for lookup; prototypes use 5 or 50 labelled train examples per class",
        "training_cohort": "pure", "validation_cohort": "pure",
        "excluded_policy": "discard conflicting-label image bytes, deduplicate identical PNGs within each split",
        "full_cohort": True,
        "gp_labels_never_used_to_train_or_modify_backbone": True,
        "code_sha256": sha(Path(__file__)),
    }
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != config:
            raise ValueError("Resume config mismatch; output must match manifest, checkpoint and parameters")
    else:
        atomic_json(config_path, config)
    atomic_json(run_dir / "population_summary.json", checks)
    write_csv(run_dir / "population_train.csv", train, list(train[0]))
    write_csv(run_dir / "population_validation.csv", val, list(val[0]))
    write_csv(run_dir / "excluded_rows.csv", excluded, ["split", "sample_id", "reason"])
    write_csv(run_dir / "support_selection_audit.csv", support_audit, list(support_audit[0]))
    write_csv(run_dir / "support_diversity_summary.csv", support_summary, list(support_summary[0]))
    print("Saved support selection audits before GPU extraction", flush=True)

    features_root = run_dir / "features"
    arrays, progress = prepare_feature_cache(features_root, train, val, config, resume=args.resume)
    if any(progress[split] < len(populations[split]) for split in populations):
        tf, extractor, mode = load_frozen_extractor(checkpoint)
        atomic_json(run_dir / "model_load.json", {"load_mode": mode, "tensorflow": tf.__version__,
                                                   "checkpoint_sha256": checkpoint_sha, "stages": STAGES})
        # Optional expensive audit: verify the real image bytes against the manifest.
        if not args.skip_cache_sha:
            from modern_pca.gleason import cache_path
            for split, rows in populations.items():
                for i in range(progress[split], len(rows)):
                    row = rows[i]
                    if sha(cache_path(args.cache, row)) != row["cache_sha256"]:
                        raise ValueError("Cache image digest mismatch: " + row["sample_id"])
                print("Cache SHA256 verified for unextracted", split, flush=True)
        for split, rows in (("train", train), ("validation", val)):
            if progress[split] < len(rows):
                extract_split(tf, extractor, rows, args.cache, arrays[split], split, progress[split],
                              args.batch_size, features_root)
        del extractor

    # Running again with --resume skips full NASNet extraction; reruns readout
    # using the same frozen support manifests and cached embeddings.
    print("All features ready; starting CPU-readable classification", flush=True)
    summary = evaluate_all(run_dir, arrays, populations, selected, seeds,
                           args.nn_query_block, args.nn_bank_block, args.nn_device)
    results = {"complete": True, "completed_utc": utc(), "config": config,
               "population": checks, "nine_conditions": summary,
               "validation_is_internal_not_external_test": True,
               "patient_independence_not_proven_if_missing_patient_ids": True}
    atomic_json(run_dir / "result.json", results)
    hashes = {}
    # Hash every generated file except this inventory and ephemeral progress.
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.name != "artifact_sha256.json":
            hashes[p.relative_to(run_dir).as_posix()] = {"sha256": sha(p), "bytes": p.stat().st_size}
    atomic_json(run_dir / "artifact_sha256.json", hashes)
    print("SUCCESS: nine conditions completed; output:", run_dir, flush=True)


def cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", type=Path, required=True, help="Full frozen PANDA PREPARED_DIR")
    ap.add_argument("--cache", type=Path, required=True, help="PREPARED_DIR/cache")
    ap.add_argument("--checkpoint", type=Path, help="Verified released binary epoch14.keras")
    ap.add_argument("--output", type=Path, help="New result directory")
    ap.add_argument("--preflight", action="store_true", help="Manifest + support selection check, no GPU/model")
    ap.add_argument("--model-smoke", action="store_true", help="Run one real PANDA image through frozen GPU backbone; no result files")
    ap.add_argument("--resume", action="store_true", help="Resume same features/config after PBS timeout")
    ap.add_argument("--skip-cache-sha", action="store_true", help="Skip per-PNG SHA audit (NOT recommended)")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--nn-device", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--nn-query-block", type=int, default=128)
    ap.add_argument("--nn-bank-block", type=int, default=2048)
    args = ap.parse_args()
    if args.batch_size < 1 or args.repeats < 1 or args.nn_query_block < 1 or args.nn_bank_block < 1:
        ap.error("batch-size, repeats and NN blocks must be positive")
    if not args.preflight and args.checkpoint is None:
        ap.error("--checkpoint is required unless --preflight")
    if not (args.preflight or args.model_smoke) and args.output is None:
        ap.error("--output is required for the full experiment")
    run(args)


if __name__ == "__main__":
    cli()
