from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data import build_datasets


def c8_views(images: tf.Tensor) -> list[tf.Tensor]:
    r90 = tf.image.rot90(images, 1)
    r180 = tf.image.rot90(images, 2)
    r270 = tf.image.rot90(images, 3)
    return [
        images,
        r90,
        r180,
        r270,
        tf.image.flip_up_down(r90),
        tf.image.flip_up_down(r270),
        tf.image.flip_up_down(images),
        tf.image.flip_left_right(images),
    ]


def predict(model, dataset, use_c8: bool):
    labels, probabilities = [], []
    started = time.perf_counter()
    for images, targets in dataset:
        if use_c8:
            stack = tf.stack([model(view, training=False) for view in c8_views(images)], axis=0)
            preds = tf.sort(stack, axis=0)[3:5]
            preds = tf.reduce_mean(preds, axis=0)
        else:
            preds = model(images, training=False)
        labels.extend(targets.numpy().tolist())
        probabilities.extend(preds.numpy().tolist())
    return np.asarray(labels), np.asarray(probabilities), time.perf_counter() - started


def metrics(labels, probabilities):
    predicted = probabilities.argmax(axis=1)
    result = {
        "accuracy": accuracy_score(labels, predicted),
        "balanced_accuracy": balanced_accuracy_score(labels, predicted),
        "precision_macro": precision_score(labels, predicted, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predicted, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, predicted, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predicted).tolist(),
    }
    try:
        result["roc_auc"] = (
            roc_auc_score(labels, probabilities[:, 1])
            if probabilities.shape[1] == 2
            else roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
        )
    except ValueError:
        result["roc_auc"] = None
    return result


def main():
    parser = argparse.ArgumentParser(description="C1/C8 complete test-split validation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--stain-reference")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data = build_datasets(
        args.data_root, args.image_size, args.batch_size, max_samples=args.max_samples,
        stain_reference=args.stain_reference,
    )
    model = tf.keras.models.load_model(args.model)
    report = {"class_names": data.class_names, "counts": data.counts}
    for name, use_c8 in (("C1", False), ("C8", True)):
        labels, probabilities, elapsed = predict(model, data.test, use_c8)
        report[name] = metrics(labels, probabilities)
        report[name]["elapsed_seconds"] = elapsed
        report[name]["images_per_second"] = len(labels) / elapsed
    (output / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
