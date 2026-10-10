import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

EXPECTED_SHA = "433dd48aae800c659d1cba092af2859499bb804d2228ff978a6e9c9602b2adde"

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    from modern_pca import reimplementation as r
    from modern_pca.evaluate_paper import calculate_binary_metrics

    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if digest(args.model) != EXPECTED_SHA:
        raise RuntimeError("Checkpoint SHA-256 mismatch")

    manifest = root / "manifests/singapore_test_eval.csv"
    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 4261
    if args.limit:
        rows = rows[:args.limit]

    tf.keras.mixed_precision.set_global_policy("float32")
    tf.config.experimental.enable_tensor_float_32_execution(False)

    model = tf.keras.models.load_model(args.model, compile=False)
    r.validate_model(model)
    model.trainable = False

    normalizers = r.normalizers(r.REFERENCE)
    print("Model and normalizers loaded", flush=True)

    predictions = []
    start = time.time()

    for i, row in enumerate(rows, 1):
        path = root / row["image_path"]
        sample = {
            "sample_id": row["sample_id"],
            "file_sha256": digest(path),
        }

        x = r.preprocess_legacy(sample, path, normalizers)
        probabilities = np.asarray(
            model(np.expand_dims(x, 0), training=False).numpy()[0],
            dtype=np.float64,
        )

        if probabilities.shape != (2,):
            raise RuntimeError("Expected two-class softmax output")
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Non-finite probabilities")
        if abs(probabilities.sum() - 1.0) > 1e-5:
            raise RuntimeError("Invalid probability sum")

        score = float(probabilities[1])
        predicted = int(score > 0.5)

        predictions.append({
            "sample_id": row["sample_id"],
            "slide_id": row["slide_id"],
            "label": int(row["label"]),
            "p_benign": float(probabilities[0]),
            "p_tumour": score,
            "prediction": predicted,
        })

        if i % 10 == 0 or i == len(rows):
            print(
                f"{i}/{len(rows)} processed "
                f"({time.time() - start:.1f}s elapsed)",
                flush=True,
            )

    labels = [r["label"] for r in predictions]
    scores = [r["p_tumour"] for r in predictions]

    metrics = calculate_binary_metrics(labels, scores)

    with (output / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)

    summary = {
        "dataset": "Singapore gland classification",
        "split": "test",
        "checkpoint_sha256": EXPECTED_SHA,
        "num_patches": len(predictions),
        "elapsed_seconds": time.time() - start,
        "metrics": metrics,
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
