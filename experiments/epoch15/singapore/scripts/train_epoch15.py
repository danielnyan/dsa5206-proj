import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from modern_pca import reimplementation as r
from modern_pca import vanda_distributed_training as t

BASE = Path("/scratch/e1536052/DSA5206")
DATA = BASE / "singapore_external"
CACHE = DATA / "cache_epoch15/full11020"
RUN14 = BASE / "vanda_runs/nasnet_production_14epoch"

BATCH = 100
SEED = 42
EPOCH = 15

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=11020)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    if not 1 <= args.limit <= 11020:
        raise ValueError("Invalid sample limit")

    r.gpu_setup("float32")
    if len(tf.config.list_physical_devices("GPU")) != 2:
        raise RuntimeError("Two GPUs required")

    strategy = tf.distribute.MirroredStrategy()
    print("REPLICAS:", strategy.num_replicas_in_sync, flush=True)

    verification = json.loads(
        (CACHE / "verification.json").read_text()
    )
    if verification["status"] != "PASS" or verification["samples"] != 11020:
        raise RuntimeError("Unverified Singapore cache")

    for name, key in [
        ("images.npy", "images_sha256"),
        ("labels.npy", "labels_sha256"),
        ("samples.csv", "samples_sha256"),
    ]:
        if t.digest(CACHE / name) != verification[key]:
            raise RuntimeError(f"Cache checksum mismatch: {name}")

    images = np.load(CACHE / "images.npy", mmap_mode="r")
    labels = np.load(CACHE / "labels.npy", mmap_mode="r")
    with (CACHE / "samples.csv").open(newline="") as f:
        samples = list(csv.DictReader(f))

    if images.shape != (11020, 350, 350, 3):
        raise RuntimeError("Unexpected image cache shape")
    if images.dtype != np.float32:
        raise RuntimeError("Unexpected image dtype")
    if labels.shape != (11020,) or len(samples) != 11020:
        raise RuntimeError("Cache metadata mismatch")
    if sum(int(x == 0) for x in labels) != 4337:
        raise RuntimeError("Benign count mismatch")
    if sum(int(x == 1) for x in labels) != 6683:
        raise RuntimeError("Tumour count mismatch")
    if any(int(s["label"]) != int(labels[i])
           for i, s in enumerate(samples)):
        raise RuntimeError("Sample-label mismatch")
    if len({s["sample_id"] for s in samples}) != 11020:
        raise RuntimeError("Duplicate sample ID")

    print("CACHE VERIFIED: 11020 samples", flush=True)

    folder = RUN14 / "epoch_14_resume"
    meta = json.loads((folder / "manifest.json").read_text())

    if (meta["completed_epochs"] != 14
        or meta["boundary"] != "normal_conv_1_1"
        or meta["rate"] != 1e-6):
        raise RuntimeError("Unexpected epoch-14 checkpoint")

    t.verify_checkpoint(folder, meta["signature"])

    model = t.initial_model(strategy)
    optimizer, count = t.configure_optimizer(
        strategy, model, meta["boundary"], meta["rate"]
    )
    t.restore_checkpoint(
        RUN14, meta, model, optimizer, meta["signature"]
    )
    if int(optimizer.iterations.numpy()) != 1167:
        raise RuntimeError("Unexpected initial optimizer iterations")

    print("RESTORED: 1167 Adam iterations", flush=True)
    print("TRAINABLE PARAMETERS:", count, flush=True)

    order = np.random.default_rng(SEED).permutation(11020)
    if args.limit != 11020:
        order = order[:args.limit]

    train_fn = t.make_train_fn(strategy, model, optimizer)
    start = time.perf_counter()
    weighted_loss = 0.0
    processed = 0

    for offset in range(0, len(order), BATCH):
        indices = order[offset:offset + BATCH]
        x = np.asarray(images[indices], dtype=np.float32).copy()
        y = np.asarray(labels[indices], dtype=np.int64)

        if not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise RuntimeError("Invalid image batch")

        for j, index in enumerate(indices):
            x[j] = np.asarray(
                r.augment(x[j], samples[int(index)]["sample_id"], EPOCH),
                dtype=np.float32,
            )

        n = len(indices)
        loss = float(train_fn(
            t.distribute(strategy, x, n),
            t.distribute(strategy, y, n),
            tf.constant(n, tf.int32),
        ).numpy())

        if not np.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")

        processed += n
        weighted_loss += loss * n

        print(
            f"TRAIN {processed}/{len(order)} "
            f"updates={int(optimizer.iterations.numpy())} "
            f"loss={weighted_loss / processed:.6f} "
            f"elapsed_min={(time.perf_counter()-start)/60:.2f}",
            flush=True,
        )

    expected_updates = (len(order) + BATCH - 1) // BATCH
    expected_iterations = 1167 + expected_updates

    if processed != len(order):
        raise RuntimeError("Incomplete epoch")
    if int(optimizer.iterations.numpy()) != expected_iterations:
        raise RuntimeError("Unexpected optimizer update count")

    print("TRAINING COMPLETE", flush=True)

    # Smoke test deliberately does not save a model.
    if args.limit != 11020:
        print("PASS: Gradient-update smoke test", flush=True)
        return

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    checkpoint = output / "epoch15_singapore.keras"
    model.save(checkpoint)

    summary = {
        "experiment": "Singapore all-patch epoch-15 fine-tuning",
        "source_checkpoint": str(RUN14 / "epoch_14_resume"),
        "source_optimizer_iterations": 1167,
        "samples": processed,
        "updates": expected_updates,
        "final_optimizer_iterations": expected_iterations,
        "learning_rate": 1e-6,
        "trainable_boundary": meta["boundary"],
        "batch_size": BATCH,
        "shuffle_seed": SEED,
        "augmentation_epoch": EPOCH,
        "mean_training_loss": weighted_loss / processed,
        "checkpoint_sha256": t.digest(checkpoint),
        "elapsed_seconds": time.perf_counter() - start,
    }
    t.write_json_atomic(output / "training_summary.json", summary)
    print("PASS: Epoch-15 checkpoint saved", flush=True)

if __name__ == "__main__":
    main()
