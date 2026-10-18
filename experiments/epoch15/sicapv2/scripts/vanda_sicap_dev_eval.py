"""Patient-separated SICAPv2 development evaluation; no training or augmentation."""

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


BASE = Path("/scratch/e1536052/DSA5206")
DEV = Path("reports/stage2j_followup/sicap_dev_patients.csv")
EXPECTED_DEV_SHA = "660273c4ce1959b77c4b35e39bbe713309d14f921fe62f8012c2f3334e43245a"

MODELS = {
    'partial': (
        BASE / 'vanda_runs/sicap_partial_epoch15_lr1e7/epoch15_partial.keras',
        'f28c7cbd6b874a4287972913127a361d7bbc8d2bcd8cf03c72747d8589c707df',
    ),
    "epoch14": (
        BASE / "vanda_runs/nasnet_production_14epoch/epoch14.keras",
        "433dd48aae800c659d1cba092af2859499bb804d2228ff978a6e9c9602b2adde",
    ),
    "head_only": (
        BASE / "vanda_runs/sicap_head_only_epoch15/epoch15_head_only.keras",
        "4d5b2127484f1e2c37ff33c3bd7ca451f77984d47b4f43b09f9710c6fd64dfd7",
    ),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def auc_rank(y, scores):
    """ROC-AUC using average ranks for tied scores."""
    order = np.argsort(scores, kind="mergesort")
    ordered = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and ordered[j] == ordered[i]:
            j += 1
        ranks[order[i:j]] = ((i + 1) + j) / 2.0
        i = j
    positives = y == 1
    n1 = int(positives.sum())
    n0 = len(y) - n1
    return float((ranks[positives].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics(y, scores):
    pred = scores > 0.5
    tp = int(((y == 1) & pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": (tp + tn) / len(y),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": 2 * tp / (2 * tp + fp + fn),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "roc_auc": auc_rank(y, scores),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True, choices=sorted(MODELS))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError("Output already exists; refusing overwrite")
    if sha256(DEV) != EXPECTED_DEV_SHA:
        raise RuntimeError("Development manifest hash mismatch")

    model_path, expected_model_sha = MODELS[args.model_key]
    if sha256(model_path) != expected_model_sha:
        raise RuntimeError("Model SHA256 mismatch")

    from modern_pca import reimplementation as r
    import tensorflow as tf

    with DEV.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 1833:
        raise RuntimeError("Unexpected development sample count")

    y = np.array([int(x["ground_truth_code"]) for x in rows], dtype=np.int32)
    if int((y == 0).sum()) != 585 or int((y == 1).sum()) != 1248:
        raise RuntimeError("Development class counts mismatch")

    if len({x["sample_id"] for x in rows}) != len(rows):
        raise RuntimeError("Duplicate development sample ID")

    train_patients = set()
    with Path("reports/stage2j_followup/sicap_train_patients.csv").open(newline="") as f:
        for x in csv.DictReader(f):
            train_patients.add(x["patient_id"])
    dev_patients = {x["patient_id"] for x in rows}
    if train_patients & dev_patients:
        raise RuntimeError("Patient overlap between training and development")

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) != 1:
        raise RuntimeError(f"Expected one visible GPU, found {len(gpus)}")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.keras.mixed_precision.set_global_policy("float32")

    model = tf.keras.models.load_model(model_path, compile=False)
    r.validate_model(model)
    model.trainable = False

    @tf.function(reduce_retracing=True)
    def infer(x):
        return model(x, training=False)

    reference = r.normalizers(r.REFERENCE)
    image_root = BASE / "sicap_transfer/images"
    scores = []
    output_rows = []

    print(
        f"SICAP DEV START model={args.model_key} "
        f"samples={len(rows)} model_sha={expected_model_sha}",
        flush=True,
    )

    for offset in range(0, len(rows), 32):
        group = rows[offset:offset + 32]
        images = []

        for row in group:
            path = image_root / row["filename"]
            if not path.is_file():
                raise FileNotFoundError(str(path))
            pixels = r.preprocess_legacy(row, path, reference)
            images.append(np.asarray(pixels, dtype=np.float32))

        batch = np.stack(images)
        if batch.shape != (len(group), 350, 350, 3):
            raise RuntimeError("Preprocessed image shape mismatch")
        if not np.isfinite(batch).all():
            raise RuntimeError("Nonfinite preprocessed images")

        probabilities = infer(tf.convert_to_tensor(batch)).numpy()
        if probabilities.shape != (len(group), 2):
            raise RuntimeError("Invalid prediction shape")
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Nonfinite predictions")

        for row, probability in zip(group, probabilities):
            score = float(probability[1])
            scores.append(score)
            output_rows.append({
                "sample_id": row["sample_id"],
                "patient_id": row["patient_id"],
                "ground_truth_code": row["ground_truth_code"],
                "p_benign": float(probability[0]),
                "p_tumour": score,
                "predicted_binary_code": int(score > 0.5),
            })

        done = offset + len(group)
        if done % 320 < 32 or done == len(rows):
            print(f"SICAP DEV {done}/{len(rows)}", flush=True)

    scores = np.array(scores, dtype=np.float64)
    result_metrics = metrics(y, scores)

    if sha256(model_path) != expected_model_sha:
        raise RuntimeError("Model changed during evaluation")

    args.output.mkdir(parents=True, exist_ok=False)

    with (args.output / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    result = {
        "purpose": "SICAPv2_patient_disjoint_development_evaluation",
        "model_key": args.model_key,
        "model_sha256": expected_model_sha,
        "development_manifest_sha256": EXPECTED_DEV_SHA,
        "samples": len(rows),
        "patients": len(dev_patients),
        "threshold": "p_tumour > 0.5",
        "augmentation": False,
        "training_performed": False,
        "metrics": result_metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }

    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")

    hashes = {
        filename: sha256(args.output / filename)
        for filename in ("metrics.json", "predictions.csv")
    }
    (args.output / "run_complete.json").write_text(
        json.dumps({"status": "PASS", "artifacts_sha256": hashes}, indent=2) + "\n"
    )

    print("=== SICAP DEVELOPMENT EVALUATION PASS ===", flush=True)
    print(json.dumps(result_metrics, indent=2), flush=True)
    print("Results:", args.output, flush=True)


if __name__ == "__main__":
    main()
