from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import tensorflow as tf

from .data import build_datasets
from .models import build_classifier, compile_classifier, configure_fine_tuning


PAPER_SCHEDULE = [
    ("normal_conv_1_17", 7, 1e-5),
    ("normal_conv_1_14", 1, 1e-6),
    ("normal_conv_1_11", 1, 1e-6),
    ("normal_conv_1_9", 1, 1e-6),
    ("normal_conv_1_7", 1, 1e-6),
    ("normal_conv_1_5", 1, 1e-6),
    ("normal_conv_1_3", 1, 1e-6),
    ("normal_conv_1_1", 1, 1e-6),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Modern Tolkach et al. patch classifier training")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--architecture", default="nasnetlarge", choices=["nasnetlarge", "resnet50", "efficientnetv2b0"])
    parser.add_argument("--head", default="original", choices=["original", "gap"])
    parser.add_argument("--weights", default="imagenet", choices=["imagenet", "none"])
    parser.add_argument("--schedule", default="paper", choices=["paper", "decision"])
    parser.add_argument("--image-size", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stain-reference")
    return parser.parse_args()


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data = build_datasets(
        args.data_root, args.image_size, args.batch_size, seed=args.seed,
        max_samples=args.max_samples, stain_reference=args.stain_reference,
    )
    model = build_classifier(
        len(data.class_names), args.architecture, args.image_size, args.head,
        None if args.weights == "none" else args.weights,
    )
    if args.schedule == "paper":
        if args.architecture != "nasnetlarge":
            raise ValueError("The paper schedule is specific to NASNetLarge; use --schedule decision")
        schedule = PAPER_SCHEDULE
    else:
        boundary = "normal_conv_1_17" if args.architecture == "nasnetlarge" else None
        schedule = [(boundary, 1, 1e-5)]
    history_rows = []
    started = time.perf_counter()
    for stage, (boundary, epochs, learning_rate) in enumerate(schedule, 1):
        configure_fine_tuning(model, boundary)
        compile_classifier(model, learning_rate)
        stage_started = time.perf_counter()
        history = model.fit(
            data.train,
            validation_data=data.validation,
            epochs=epochs,
            callbacks=[tf.keras.callbacks.TerminateOnNaN()],
            verbose=2,
        )
        stage_seconds = time.perf_counter() - stage_started
        for epoch in range(epochs):
            row = {key: values[epoch] for key, values in history.history.items()}
            row.update(stage=stage, boundary=boundary or "all", stage_seconds=stage_seconds)
            history_rows.append(row)
        # Keep a reloadable recovery point without storing eight ~GB-scale copies
        # of the paper's unusually large Flatten/Dense head.
        model.save(output / "latest_stage.keras", overwrite=True)
    test_metrics = model.evaluate(data.test, return_dict=True, verbose=2)
    elapsed = time.perf_counter() - started
    model.save(output / "final.keras")
    pd.DataFrame(history_rows).to_csv(output / "history.csv", index=False)
    summary = {
        "architecture": args.architecture,
        "head": args.head,
        "schedule": args.schedule,
        "class_names": data.class_names,
        "counts": data.counts,
        "test_metrics": {key: float(value) for key, value in test_metrics.items()},
        "elapsed_seconds": elapsed,
        "tensorflow": tf.__version__,
        "gpus": [item.name for item in tf.config.list_physical_devices("GPU")],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
