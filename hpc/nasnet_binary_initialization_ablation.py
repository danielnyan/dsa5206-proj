"""Binary VAL1 NASNetLarge scratch / ImageNet initialization ablation.

Requires existing repo modules and its frozen, verified Stage-2G VAL1 cache.
Train:  python -m hpc.nasnet_binary_initialization_ablation train --cache CACHE --output RUN --initialization scratch --learning-rate 1e-4 --max-epochs 1
Resume: same with --max-epochs 14; full optimizer+BN state preserved per epoch.
Test:   python -m hpc.nasnet_binary_initialization_ablation test --model-dir RUN --extracted VAL2_EXTRACTED --output RUN/test_val2 --final-test

NASNetLarge has random initialized parameters when weights=None; 'no initialization' is not literal.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

EXPECTED = {"train": 116655, "internal_validation": 29164}
CLASSES = ("benign", "tumour")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def setup_gpu(seed):
    import tensorflow as tf
    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError("GPU allocation is required")
    # Require exactly one visible device: accumulation works within one replica.
    if len(devices) != 1:
        raise RuntimeError(f"Expected exactly one visible GPU, found {len(devices)}. Set CUDA_VISIBLE_DEVICES in PBS.")
    tf.config.experimental.set_memory_growth(devices[0], True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.keras.utils.set_random_seed(seed)
    # Preserve the same deterministic setting as the native reimplementation.
    tf.config.experimental.enable_op_determinism()
    return tf


def model_and_optimizer(initialization, lr):
    import tensorflow as tf
    from modern_pca.models import build_classifier
    from modern_pca import reimplementation as r
    model = build_classifier(2, weights=None if initialization == "scratch" else "imagenet",
                             image_size=350, head="original", backbone_training=None)
    model.trainable = True
    model.backbone.trainable = True
    for layer in model.backbone.layers:
        layer.trainable = True
    r.validate_model(model)
    assert all(layer.trainable for layer in model.backbone.layers), "Not all backbone layers trainable"
    assert model.count_params() == 209812820
    opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=.9, beta_2=.999,
                                   epsilon=1e-7, amsgrad=False)
    opt.build(model.trainable_variables)
    return model, opt


def experiment_signature(args):
    from modern_pca import preprocessing_cache as c
    return dict(protocol="binary_val1_full_unfreeze_v1", initialization=args.initialization,
                seed=args.seed, learning_rate=args.learning_rate, physical_batch=args.batch_size,
                effective_batch=args.effective_batch, precision="float32", tf32=False,
                training_mode=True, batchnorm="trainable; microbatch training stats",
                train_count=EXPECTED["train"], val_count=EXPECTED["internal_validation"],
                cache_catalog_sha256=sha(args.cache / "cache.json"), split_hashes=c.SPLIT_HASHES,
                python_sha256=sha(Path(__file__)), model_parameters=209812820,
                head="Flatten-Dense256ReLU-Dense2Softmax", scheduled_epochs=14,
                train_augmentation="same deterministic horizontal+vertical flips as verified VAL1 cache native trainer",
                validation_selection="frozen VAL1 internal patch-level split; VAL2 excluded")


def ckpt_dir(out, epoch):
    return out / f"epoch_{epoch:02d}"


def save_epoch(out, epoch, model, opt, history, signature):
    import tensorflow as tf
    dest = ckpt_dir(out, epoch)
    if dest.exists():
        raise FileExistsError(f"Would overwrite existing epoch: {dest}")
    staging = out / (f".epoch_{epoch:02d}_" + uuid.uuid4().hex)
    staging.mkdir()
    try:
        tf.train.Checkpoint(model=model, optimizer=opt).write(str(staging / "weights"))
        inventory = {p.name: sha(p) for p in staging.glob("weights.*")}
        if not any(p.endswith(".index") for p in inventory) or not any(".data-" in p for p in inventory):
            raise RuntimeError("Incomplete checkpoint")
        atomic_json(staging / "checkpoint.json", dict(epoch=epoch, signature=signature,
                   history=history, optimizer_iterations=int(opt.iterations.numpy()), files_sha256=inventory))
        staging.rename(dest)  # same-FS publish after all files exist
        atomic_json(out / "progress.json", dict(completed_epochs=epoch, history=history,
                                                checkpoint=str(dest.name), signature=signature))
        # Retain initial pilot and newest two checkpoints, not 14 multi-GB sets.
        for old in range(2, epoch-1):
            p = ckpt_dir(out, old)
            if p.is_dir():
                shutil.rmtree(p)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def restore_latest(out, model, opt, signature):
    import tensorflow as tf
    progress = out / "progress.json"
    if not progress.exists():
        if any(out.iterdir()):
            raise RuntimeError("Output exists but has no committed progress.json; inspect before retry")
        return 0, []
    info = json.loads(progress.read_text())
    if info["signature"] != signature:
        raise ValueError("Training code/data/hyperparameter signature changed; resume rejected")
    epoch = info["completed_epochs"]
    folder = ckpt_dir(out, epoch)
    manifest = json.loads((folder / "checkpoint.json").read_text())
    if manifest["signature"] != signature or manifest["epoch"] != epoch:
        raise ValueError("Checkpoint identity mismatch")
    for name, expected in manifest["files_sha256"].items():
        if sha(folder / name) != expected:
            raise ValueError(f"Checkpoint corrupted: {name}")
    tf.train.Checkpoint(model=model, optimizer=opt).read(str(folder / "weights")).assert_consumed()
    if int(opt.iterations.numpy()) != manifest["optimizer_iterations"]:
        raise ValueError("Optimizer restoration failed")
    if manifest["history"] != info["history"]:
        raise ValueError("Progress/history disagree")
    return epoch, info["history"]


def new_train_functions(model, opt):
    import tensorflow as tf
    variables = list(model.trainable_variables)
    buffers = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False)
               for v in variables]

    @tf.function(reduce_retracing=True)
    def microstep(images, labels):
        with tf.GradientTape() as tape:
            probs = model(images, training=True)
            losses = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
            loss_sum = tf.reduce_sum(losses)
        grads = tape.gradient(loss_sum, variables)
        tf.debugging.assert_all_finite(loss_sum, "Nonfinite loss")
        for grad, buf in zip(grads, buffers):
            if grad is None:
                raise RuntimeError("Disconnected parameter gradient")
            if isinstance(grad, tf.IndexedSlices):
                grad = tf.convert_to_tensor(grad)
            tf.debugging.assert_all_finite(grad, "Nonfinite gradient")
            buf.assign_add(tf.cast(grad, tf.float32))
        correct = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(probs, axis=-1, output_type=tf.int64),
                                                  tf.cast(labels, tf.int64)), tf.int32))
        return loss_sum, correct

    @tf.function(reduce_retracing=True)
    def update(n):
        count = tf.cast(n, tf.float32)
        opt.apply_gradients([(buf / count, var) for buf, var in zip(buffers, variables)])
        for buf in buffers:
            buf.assign(tf.zeros_like(buf))
        return opt.iterations
    return microstep, update


def train_one_epoch(reader, epoch, model, opt, batch, effective, log_every):
    from modern_pca import preprocessing_cache as c
    indices = reader.indices("train", epoch)
    if len(indices) != EXPECTED["train"] or len(set(indices)) != len(indices):
        raise RuntimeError("Train cohort changed")
    microstep, update = new_train_functions(model, opt)
    ds = c.dataset(reader, indices, batch_size=batch, parallel=2, prefetch=1,
                   augment=True, epoch=epoch)
    seen = pending = steps = updates = correct = 0
    loss_total = 0.0
    started = time.monotonic()
    for images, labels, idx in ds:
        n = int(labels.shape[0])
        if idx.numpy().tolist() != indices[seen:seen+n]:
            raise ValueError("Training sample order mismatch")
        loss_sum, n_correct = microstep(images, labels)
        loss_total += float(loss_sum.numpy())
        correct += int(n_correct.numpy())
        seen += n; pending += n; steps += 1
        if pending >= effective or seen == EXPECTED["train"]:
            # Effective batch exactly 100 except final 55; flush at epoch boundary.
            if pending > effective:
                raise ValueError("Effective batch must be divisible by physical batch")
            update(tf.constant(pending, dtype=__import__('tensorflow').int32))
            updates += 1; pending = 0
        if steps % log_every == 0 or seen == EXPECTED["train"]:
            print(f"TRAIN epoch={epoch} images={seen}/{EXPECTED['train']} "
                  f"loss={loss_total/seen:.6f} accuracy={correct/seen:.5f} "
                  f"minutes={(time.monotonic()-started)/60:.1f}", flush=True)
    if seen != EXPECTED["train"] or pending != 0:
        raise ValueError("Incomplete epoch")
    return dict(loss=loss_total/seen, accuracy=correct/seen, samples=seen, updates=updates,
                wall_seconds=time.monotonic()-started)


def eval_val1(reader, model, batch):
    from modern_pca import preprocessing_cache as c
    import tensorflow as tf
    indices = reader.indices("internal_validation")
    if len(indices) != EXPECTED["internal_validation"]:
        raise ValueError("Validation population mismatch")

    @tf.function(reduce_retracing=True)
    def predict(images):
        return model(images, training=False)

    total_loss = correct = processed = 0
    scores = []
    targets = []
    started = time.monotonic()
    for images, labels, idx in c.dataset(reader, indices, batch_size=batch, parallel=2,
                                        prefetch=1, augment=False, epoch=1):
        n = int(labels.shape[0])
        if idx.numpy().tolist() != indices[processed:processed+n]:
            raise ValueError("Validation identity mismatch")
        probs = predict(images)
        tf.debugging.assert_all_finite(probs, "Nonfinite validation probabilities")
        losses = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
        total_loss += float(tf.reduce_sum(losses).numpy())
        label = labels.numpy()
        output = probs.numpy()
        correct += int((output.argmax(axis=1) == label).sum())
        scores.extend(output[:,1].tolist())
        targets.extend(label.tolist())
        processed += n
    if processed != EXPECTED["internal_validation"]:
        raise ValueError("Incomplete validation")
    from modern_pca.evaluate_paper import calculate_binary_metrics
    metrics = calculate_binary_metrics(targets, scores)
    return dict(loss=total_loss/processed, accuracy=correct/processed, samples=processed,
                metrics=json.loads(json.dumps(metrics, default=float)),
                wall_seconds=time.monotonic()-started)


def run_train(args):
    from modern_pca.vanda_cache_consumer import open_verified_cache
    from modern_pca import reimplementation as r
    tf = setup_gpu(args.seed)
    if args.batch_size <= 0 or args.effective_batch % args.batch_size:
        raise ValueError("Effective batch must be divisible by physical batch")
    if not 1 <= args.max_epochs <= 14:
        raise ValueError("max-epochs outside 1..14")
    if args.output.exists() and not args.output.is_dir():
        raise ValueError("Output is not directory")
    args.output.mkdir(parents=True, exist_ok=True)
    signature = experiment_signature(args)
    print(f"MODE=train init={args.initialization} LR={args.learning_rate} signature={signature}", flush=True)
    reader = open_verified_cache(args.cache)
    model, opt = model_and_optimizer(args.initialization, args.learning_rate)
    beginning, history = restore_latest(args.output, model, opt, signature)
    for epoch in range(beginning+1, args.max_epochs+1):
        print(f"EPOCH START {epoch}/14 fully trainable; no layers frozen", flush=True)
        train_stats = train_one_epoch(reader, epoch, model, opt, args.batch_size,
                                      args.effective_batch, args.log_every)
        validation = eval_val1(reader, model, args.validation_batch)
        row = dict(epoch=epoch, train=train_stats, validation=validation)
        history.append(row)
        save_epoch(args.output, epoch, model, opt, history, signature)
        with (args.output / "loss_curves.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","train_accuracy","val_accuracy",
                                                    "train_seconds","val_seconds"])
            writer.writeheader()
            for h in history:
                writer.writerow(dict(epoch=h["epoch"],train_loss=h["train"]["loss"],
                                 val_loss=h["validation"]["loss"],train_accuracy=h["train"]["accuracy"],
                                 val_accuracy=h["validation"]["accuracy"],
                                 train_seconds=h["train"]["wall_seconds"],
                                 val_seconds=h["validation"]["wall_seconds"]))
        print(f"EPOCH COMPLETE {epoch} train_loss={train_stats['loss']:.5f} "
              f"val_loss={validation['loss']:.5f} val_accuracy={validation['accuracy']:.5f}", flush=True)
    if args.max_epochs == 14:
        path = args.output / "epoch14_scratch.keras" if args.initialization == "scratch" else args.output / "epoch14_imagenet_unfrozen.keras"
        if not path.exists():
            model.save(path)
        atomic_json(args.output / "final.json", dict(complete=True, checkpoint=path.name,
                   checkpoint_sha256=sha(path), completed_epochs=14,
                   signature=signature, history=history,
                   note="Final external test must be run separately; never tune on VAL2"))
        print(f"COMPLETE 14 EPOCHS export={path}", flush=True)
    else:
        print(f"PILOT COMPLETE through {args.max_epochs} epochs; resume with --max-epochs 14", flush=True)


def run_test(args):
    import numpy as np
    import tensorflow as tf
    from modern_pca import reimplementation as r
    from modern_pca.evaluate_paper import calculate_binary_metrics
    if not args.final_test:
        raise ValueError("Explicit --final-test required for VAL2")
    tf = setup_gpu(42)
    final = json.loads((args.model_dir / "final.json").read_text())
    if not final.get("complete") or final.get("completed_epochs") != 14:
        raise ValueError("Not a final 14-epoch model")
    model_path = args.model_dir / final["checkpoint"]
    if sha(model_path) != final["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    if args.output.exists():
        raise FileExistsError("Test outputs must be a fresh directory; never overwrite")
    if args.batch_size not in (1,2,4,8,16):
        raise ValueError("Invalid test batch size")
    rows = r.read_manifest(args.manifest, cohort="VAL2")
    paths = [r.image_path(x, args.extracted, final_evaluation=True) for x in rows]
    model = tf.keras.models.load_model(model_path, compile=False)
    r.validate_model(model)
    model.trainable = False
    normalizers = r.normalizers(args.stain_reference)
    args.output.mkdir(parents=True)
    start = time.monotonic(); total_loss = 0.0
    truth, scores = [], []
    import csv as csvmod
    with (args.output / "predictions.csv").open("w", newline="") as out:
        writer = csvmod.DictWriter(out, fieldnames=["sample_id","ground_truth_code","p_benign","p_tumour","predicted"])
        writer.writeheader()
        for start_idx in range(0,len(rows),args.batch_size):
            end = min(start_idx+args.batch_size,len(rows))
            items = rows[start_idx:end]
            inputs = np.stack([r.preprocess_legacy(x,p,normalizers)
                               for x,p in zip(items, paths[start_idx:end])]).astype(np.float32)
            probs = model(inputs, training=False).numpy()
            y = np.asarray([x["ground_truth_code"] for x in items], dtype=np.int64)
            if probs.shape != (len(y),2) or not np.isfinite(probs).all():
                raise ValueError("Nonfinite or wrong shape predictions")
            total_loss += float(tf.keras.losses.sparse_categorical_crossentropy(y, probs).numpy().sum())
            scores.extend(probs[:,1].tolist()); truth.extend(y.tolist())
            for row, p in zip(items,probs):
                writer.writerow(dict(sample_id=row["sample_id"],ground_truth_code=row["ground_truth_code"],
                                     p_benign=float(p[0]),p_tumour=float(p[1]),predicted=int(p[1]>.5)))
            if end % 500 < args.batch_size or end == len(rows):
                print(f"VAL2 {end}/{len(rows)} elapsed_min={(time.monotonic()-start)/60:.1f}", flush=True)
    metrics = calculate_binary_metrics(truth, scores)
    info = dict(complete=True, cohort="VAL2", samples=len(rows),
                test_cross_entropy=total_loss/len(rows), test_accuracy=metrics["accuracy"],
                metrics=json.loads(json.dumps(metrics, default=float)),
                model_sha256=final["checkpoint_sha256"],
                manifest_sha256=sha(args.manifest),
                final_epoch=14, initial_weights=final["signature"]["initialization"],
                elapsed_seconds=time.monotonic()-start,
                warning="VAL2 is test only; if already used for choices elsewhere in the project, disclose that")
    atomic_json(args.output / "test_metrics.json", info)
    print(f"FINAL VAL2 TEST cross_entropy={info['test_cross_entropy']:.6f} "
          f"accuracy={info['test_accuracy']:.5f}",flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sp = p.add_subparsers(dest="mode",required=True)
    tr = sp.add_parser("train")
    tr.add_argument("--cache",type=Path,required=True)
    tr.add_argument("--output",type=Path,required=True)
    tr.add_argument("--initialization",choices=("scratch","imagenet"),default="scratch")
    tr.add_argument("--learning-rate",type=float,default=1e-4)
    tr.add_argument("--batch-size",type=int,default=4)
    tr.add_argument("--effective-batch",type=int,default=100)
    tr.add_argument("--validation-batch",type=int,default=16)
    tr.add_argument("--seed",type=int,default=42)
    tr.add_argument("--max-epochs",type=int,default=14)
    tr.add_argument("--log-every",type=int,default=500)
    te = sp.add_parser("test")
    te.add_argument("--model-dir",type=Path,required=True)
    te.add_argument("--manifest",type=Path,default=Path("manifests/VAL2_manifest.csv"))
    te.add_argument("--extracted",type=Path,required=True)
    te.add_argument("--stain-reference",type=Path,default=Path("4_WSI_pipeline/WSI_pipeline_v6/images/standard_he_stain_small.jpg"))
    te.add_argument("--output",type=Path,required=True)
    te.add_argument("--batch-size",type=int,default=4)
    te.add_argument("--final-test",action="store_true")
    args=p.parse_args()
    if args.mode=="train":
        if not math.isfinite(args.learning_rate) or args.learning_rate<=0:
            p.error("--learning-rate must be positive finite")
        run_train(args)
    else:
        run_test(args)

if __name__=="__main__":
    main()
