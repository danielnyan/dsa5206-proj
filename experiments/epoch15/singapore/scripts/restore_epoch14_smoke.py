import json
from pathlib import Path

import tensorflow as tf

from modern_pca import reimplementation as r
from modern_pca import vanda_distributed_training as t

BASE = Path("/scratch/e1536052/DSA5206")
RUN = BASE / "vanda_runs/nasnet_production_14epoch"
FOLDER = RUN / "epoch_14_resume"

def main():
    r.gpu_setup("float32")

    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs:", len(gpus), flush=True)
    if len(gpus) != 2:
        raise RuntimeError("Expected exactly two GPUs")

    strategy = tf.distribute.MirroredStrategy()

    meta = json.loads((FOLDER / "manifest.json").read_text())

    assert meta["completed_epochs"] == 14
    assert meta["boundary"] == "normal_conv_1_1"
    assert meta["rate"] == 1e-6

    t.verify_checkpoint(FOLDER, meta["signature"])
    print("CHECKPOINT FILES VERIFIED", flush=True)

    model = t.initial_model(strategy)
    optimizer, count = t.configure_optimizer(
        strategy,
        model,
        meta["boundary"],
        meta["rate"],
    )

    t.restore_checkpoint(
        RUN,
        meta,
        model,
        optimizer,
        meta["signature"],
    )

    print("Trainable parameters:", count, flush=True)
    print("Optimizer iterations:", int(optimizer.iterations.numpy()), flush=True)
    print("PASS: Epoch-14 model and Adam state restored", flush=True)

if __name__ == "__main__":
    main()
