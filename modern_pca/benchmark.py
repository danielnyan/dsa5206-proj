from __future__ import annotations

import argparse
import json
import time

import numpy as np
import tensorflow as tf

from .data import build_datasets
from .models import build_classifier, compile_classifier, configure_fine_tuning


def main():
    parser = argparse.ArgumentParser(description="Benchmark real pathology images on the selected TensorFlow device")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--architecture", default="nasnetlarge", choices=["nasnetlarge", "resnet50", "efficientnetv2b0"])
    parser.add_argument("--head", default="original", choices=["original", "gap"])
    parser.add_argument("--weights", default="none", choices=["imagenet", "none"])
    parser.add_argument("--image-size", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--inference-steps", type=int, default=5)
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument("--stain-reference")
    args = parser.parse_args()
    data = build_datasets(
        args.data_root, args.image_size, args.batch_size, max_samples=30,
        stain_reference=args.stain_reference,
    )
    model = build_classifier(
        len(data.class_names), args.architecture, args.image_size, args.head,
        None if args.weights == "none" else args.weights,
    )
    boundary = "normal_conv_1_17" if args.architecture == "nasnetlarge" else None
    configure_fine_tuning(model, boundary)
    compile_classifier(model, 1e-5)
    batches = list(data.train.take(max(args.inference_steps, args.train_steps)))
    model(batches[0][0], training=False)
    inference_times = []
    for images, _ in batches[: args.inference_steps]:
        started = time.perf_counter()
        model(images, training=False).numpy()
        inference_times.append(time.perf_counter() - started)
    train_times = []
    for images, labels in batches[: args.train_steps]:
        started = time.perf_counter()
        model.train_on_batch(images, labels)
        train_times.append(time.perf_counter() - started)
    result = {
        "tensorflow": tf.__version__,
        "gpus": [item.name for item in tf.config.list_physical_devices("GPU")],
        "architecture": args.architecture,
        "head": args.head,
        "parameters": model.count_params(),
        "batch_size": args.batch_size,
        "inference_images_per_second": args.batch_size / float(np.median(inference_times)),
        "train_images_per_second": args.batch_size / float(np.median(train_times)) if train_times else None,
    }
    print("BENCHMARK_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
