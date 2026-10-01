"""VAL1-only constrained training. Explicit modes; never trains by default.

Run as python -m modern_pca.train_reimplementation --help. The smoke supervisor
uses a fresh process for each GPU configuration so OOM cannot poison later runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from . import reimplementation as r
from .evaluate_paper import Progress, calculate_binary_metrics


def smoke_rows(train, count=400):
    """400 distinct training images, both labels and at least one variant/class."""
    if not 400 <= count <= 1024 or any(row["cohort"] != "VAL1" for row in train):
        raise ValueError("Smoke requires 400..1024 VAL1-training images")
    selected = []
    for label in r.CLASSES:
        ranked = sorted((row for row in train if row["class_name"] == label),
                        key=lambda row: r.rank_id(row["sample_id"], "VAL1-smoke-v1|42|"))
        variants = [row for row in ranked if (row["width"], row["height"]) != (610, 612)]
        if not variants:
            raise ValueError("Smoke selection requires a native dimension variant in each class")
        chosen = [variants[0]] + [row for row in ranked if row["sample_id"] != variants[0]["sample_id"]][:count//2-1]
        selected.append(chosen)
    # Interleave classes; the variant pair is actually processed, not merely inventoried.
    result = [row for pair in zip(*selected) for row in pair]
    if len(result) != count or len({row["sample_id"] for row in result}) != count:
        raise ValueError("Insufficient unique smoke samples")
    return result


def prepare_smoke(args, train, summary):
    rows = smoke_rows(train)
    reference = r.normalizers(args.stain_reference)
    start = time.perf_counter()
    images = np.lib.format.open_memmap(args.output / "smoke_inputs.npy", mode="w+", dtype="float32", shape=(len(rows),350,350,3))
    progress = Progress(len(rows), 25, 30, r.LOG)
    for index, row in enumerate(rows):
        images[index] = r.preprocess_legacy(row, r.image_path(row, args.extracted), reference)
        progress.update(index + 1)
    images.flush()
    seconds = time.perf_counter() - start
    np.save(args.output / "smoke_labels.npy", np.array([row["ground_truth_code"] for row in rows], np.int64))
    (args.output / "smoke_samples.csv").write_bytes(r.csv_bytes(r.ordered_rows(rows)))
    config = dict(split=summary, preprocessing_seconds=seconds, preprocessing_images_per_second=len(rows)/seconds,
                  sample_ids=[row["sample_id"] for row in rows], samples=len(rows),
                  native_variants=sum((row["width"],row["height"]) != (610,612) for row in rows),
                  class_counts=dict(Counter(row["class_name"] for row in rows)),
                  cache_sha256=r.sha256_file(args.output / "smoke_inputs.npy"),
                  labels_sha256=r.sha256_file(args.output / "smoke_labels.npy"))
    r.write_json(args.output / "smoke_inputs.json", config)
    return config


def checkpoint_roundtrip(model, optimizer, output, probe):
    """Restore weights AND actual Adam slots without a second GPU-sized model.

    Checkpoints are disposable smoke artifacts; integrity of every model/optimizer
    tensor and a prediction are checked. Only the files created here are removed.
    Production uses a portable .keras model, separately tested on a small model.
    """
    import tensorflow as tf
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint.save_counter  # Materialize before collecting state.
    variables = list(model.weights) + list(optimizer.variables() if callable(optimizer.variables) else optimizer.variables)
    hashes = r.weight_hashes(variables)
    expected = model(probe, training=False).numpy()
    prefix = str(output / "disposable_checkpoint")
    saved = checkpoint.write(prefix)
    digest = {p.name: r.sha256_file(p) for p in output.glob("disposable_checkpoint.*")}
    # Perturb both a learned weight and optimizer iteration to prove restore occurs.
    model.trainable_variables[-1].assign_add(tf.ones_like(model.trainable_variables[-1]))
    optimizer.iterations.assign_add(1)
    checkpoint.read(saved).assert_consumed()
    if hashes != r.weight_hashes(variables) or not np.array_equal(expected, model(probe, training=False).numpy()):
        raise ValueError("Checkpoint reload changed state/predictions")
    for path in output.glob("disposable_checkpoint.*"):
        if path.resolve().parent != output.resolve():
            raise ValueError("Unexpected checkpoint cleanup path")
        path.unlink()
    return dict(status="PASS", model_and_adam_state=True, artifact_sha256=digest, retained=False)


def smoke_worker(args):
    import tensorflow as tf
    output = args.output
    r.logging_setup(output, args.log_level)
    result = dict(status="STARTED", boundary=args.boundary, physical_batch=args.batch_size,
                  effective_batch=100, precision=args.precision)
    try:
        config = json.loads((args.smoke_parent / "smoke_inputs.json").read_text())
        r.sha256_file(args.smoke_parent / "smoke_inputs.npy", config["cache_sha256"])
        r.sha256_file(args.smoke_parent / "smoke_labels.npy", config["labels_sha256"])
        device = r.gpu_setup(args.precision)
        r.LOG.info("GPU recognized: %s", device)
        start = time.perf_counter()
        model, counts = r.build_model(args.precision)
        result.update(model_loading_seconds=time.perf_counter()-start, parameters=model.count_params(), stage_trainable_parameters=counts)
        r.configure_stage(model, args.boundary)
        optimizer = r.adam(1e-5 if args.boundary == r.SCHEDULE[0][0] else 1e-6)
        accumulator = r.Accumulator(model, optimizer, loss_scale=128 if args.precision == "mixed_float16" else 1)
        result["metadata"] = r.metadata(config["split"], "smoke_disposable", args.batch_size, args.precision)
        result["device"] = device
        r.write_json(output / "metadata.json", result["metadata"])
        images = np.load(args.smoke_parent / "smoke_inputs.npy", mmap_mode="r")
        labels = np.load(args.smoke_parent / "smoke_labels.npy")
        frozen_before = r.weight_hashes(model.non_trainable_variables)
        train_before = r.weight_hashes(model.trainable_variables)
        times, updates, losses = [], [], []
        measured_start = None
        warmup_memory = None
        # Synchronous execution plus .numpy() at compiled function boundaries includes
        # backward/accumulation kernels, rather than timing only GPU dispatch.
        for logical in range(4):
            if logical == 1:
                try:
                    warmup_memory = tf.config.experimental.get_memory_info("GPU:0")
                    tf.config.experimental.reset_memory_stats("GPU:0")
                except (ValueError, AttributeError):
                    pass
                measured_start = time.perf_counter()
            for offset in range(logical*100, (logical+1)*100, args.batch_size):
                end = min(offset + args.batch_size, (logical+1)*100)
                start = time.perf_counter()
                x = tf.stack([r.augment(tf.convert_to_tensor(images[i]), config["sample_ids"][i], 1) for i in range(offset,end)])
                loss = accumulator.add(x, labels[offset:end])
                elapsed = time.perf_counter()-start
                if logical:
                    times.append(elapsed)
                    losses.append(loss)
                if end % 25 < args.batch_size:
                    r.LOG.info("%s batch=%d processed=%d/400",args.boundary,args.batch_size,end)
            start = time.perf_counter()
            accumulator.flush()
            update_seconds = time.perf_counter()-start
            if logical:
                updates.append(update_seconds)
            r.LOG.info("%s batch=%d logical=%d/4 loss=%.6g update=%.3fs", args.boundary, args.batch_size, logical+1, loss, update_seconds)
        measured_seconds = time.perf_counter()-measured_start
        try:
            memory = tf.config.experimental.get_memory_info("GPU:0")
        except (ValueError, AttributeError):
            memory = None
        stable = frozen_before == r.weight_hashes(model.non_trainable_variables)
        changed = train_before != r.weight_hashes(model.trainable_variables)
        if not stable or not changed or int(optimizer.iterations.numpy()) != 4:
            raise ValueError("Frozen-state/weight-update/Adam-step integrity failed")
        # Inference timing uses the same training-only sample pool, never validation/VAL2.
        inference_times = []
        for offset in range(0,100,args.batch_size):
            start = time.perf_counter()
            values = model(tf.convert_to_tensor(images[offset:offset+args.batch_size]), training=False).numpy()
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite validation probe")
            inference_times.append(time.perf_counter()-start)
        inference_seconds = sum(inference_times)
        checkpoint = checkpoint_roundtrip(model, optimizer, output, tf.convert_to_tensor(images[:1]))
        preprocessing_per_image = config["preprocessing_seconds"] / config["samples"]
        training_seconds = 116655 * (measured_seconds/300 + preprocessing_per_image)
        validation_seconds = 29164 * (inference_seconds/100 + preprocessing_per_image)
        result.update(status="PASS", measured_images=300, warmup_images=100,
                      microbatch_median_seconds=float(np.median(times)), microbatch_p95_seconds=float(np.percentile(times,95)),
                      optimizer_update_seconds=updates, optimizer_update_median_seconds=float(np.median(updates)),
                      measured_seconds=measured_seconds, images_per_second=300/measured_seconds,
                      preprocessing_images_per_second=config["preprocessing_images_per_second"],
                      validation_inference_images_per_second=100/inference_seconds,
                      estimated_epoch_training_seconds=training_seconds, estimated_validation_seconds=validation_seconds,
                      estimate_scope="Serial preprocessing plus measured compute; excludes checkpoint IO; validation timing extrapolated from VAL1-training images.",
                      memory=memory, warmup_memory=warmup_memory, finite_loss=True, finite_gradients=True, weights_updated=changed,
                      frozen_layers_stable=stable, optimizer_steps=4, checkpoint=checkpoint,
                      timing_scope="microbatch includes augmentation, transfer, forward/backward, finite checks, accumulation and synchronization; optimizer timed separately; warm-up excluded")
    except tf.errors.ResourceExhaustedError as error:
        result.update(status="OOM", error=str(error))
        try:
            result["memory_at_failure"] = tf.config.experimental.get_memory_info("GPU:0")
        except (ValueError, AttributeError):
            pass
        r.LOG.exception("GPU memory exhausted; no silent precision change")
    except Exception as error:
        result.update(status="ERROR", error=f"{type(error).__name__}: {error}")
        r.LOG.exception("Smoke configuration failed")
    r.write_json(output / "result.json", result)
    r.LOG.info("Completed UTC=%s status=%s", r.utc_now(), result["status"])
    return 0 if result["status"] == "PASS" else 2


def smoke_benchmark(args, train, summary):
    config = prepare_smoke(args, train, summary)
    results = []
    for boundary in (r.SCHEDULE[0][0], r.SCHEDULE[-1][0]):
        for precision in ("float32", "mixed_float16"):
            if precision == "mixed_float16" and not any(
                v["boundary"] == boundary and v["physical_batch"] == 1 and v["precision"] == "float32" and v["status"] == "OOM" for v in results):
                continue
            if precision == "mixed_float16":
                r.LOG.warning("Float32 batch 1 OOM: separately testing mixed_float16 fallback")
            for batch in (1,2,4):
                output = args.output / f"{boundary}_{precision}_batch{batch}"
                command = [sys.executable, "-B", "-m", "modern_pca.train_reimplementation", "--smoke-worker",
                           "--smoke-parent", str(args.output), "--output", str(output), "--boundary", boundary,
                           "--batch-size", str(batch), "--precision", precision]
                r.LOG.info("Starting configuration %s", output.name)
                with (args.output / f"{output.name}.console.log").open("w") as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                    while process.poll() is None:
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            r.LOG.info("Configuration running: %s (see its console log)", output.name)
                result_path = output / "result.json"
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                else:
                    result = dict(status="PROCESS_FAILED", boundary=boundary, physical_batch=batch, precision=precision, returncode=process.returncode)
                results.append(result)
                r.write_json(args.output / "benchmark_results.json", dict(config=config, results=results, complete=False))
                r.LOG.info("Configuration completed: %s %s", output.name, result["status"])
    viable = [b for b in (1,2,4) if all(any(v["status"] == "PASS" and v["physical_batch"] == b and
               v["precision"] == "float32" and v["boundary"] == boundary for v in results)
               for boundary in (r.SCHEDULE[0][0],r.SCHEDULE[-1][0]))]
    recommended = max(viable, key=lambda b: min(v["images_per_second"] for v in results if v["status"] == "PASS" and v["physical_batch"] == b and v["precision"] == "float32")) if viable else None
    estimate = None
    if recommended:
        chosen = [v for v in results if v["status"] == "PASS" and v["physical_batch"] == recommended and v["precision"] == "float32"]
        first, deepest = chosen
        total = lambda v: v["estimated_epoch_training_seconds"] + v["estimated_validation_seconds"]
        estimate = dict(lower_seconds=7*total(first)+7*min(total(first),total(deepest)),
                        upper_seconds=7*total(first)+7*max(total(first),total(deepest)),
                        caveat="Endpoint-stage extrapolation, not measured intermediate stages; excludes checkpoint IO and thermal/IO variation.")
    report = dict(config=config, results=results, complete=True, recommended_float32_batch=recommended,
                  estimated_14_epochs=estimate, production_decision="GO" if recommended else "NO-GO",
                  completed_at_utc=r.utc_now(), VAL2_images_accessed=False)
    r.write_json(args.output / "benchmark_results.json", report)
    return 0 if recommended else 2


def production(args, train, validation, summary):
    """Explicit future production mode: fixed 14 epochs, no selection/early stopping."""
    import tensorflow as tf
    device = r.gpu_setup()
    model, counts = r.build_model()
    meta = r.metadata(summary, "production_epoch14", args.batch_size)
    meta.update(device=device, stage_trainable_parameters=counts, completed_epochs=0)
    r.write_json(args.output / "metadata.json", meta)
    reference = r.normalizers(args.stain_reference)
    epoch, history = 0, []
    for boundary, epochs, rate in r.SCHEDULE:
        r.configure_stage(model, boundary)
        optimizer = r.adam(rate)
        accumulator = r.Accumulator(model, optimizer)
        for _ in range(epochs):
            epoch += 1
            rows = sorted(train, key=lambda row: r.rank_id(row["sample_id"], f"VAL1-epoch-v1|42|{epoch}|"))
            progress = Progress(len(rows), args.log_every, 30, r.LOG)
            start, loss_sum = time.perf_counter(), 0.
            for logical in range(0,len(rows),100):
                group = rows[logical:logical+100]
                for offset in range(0,len(group),args.batch_size):
                    batch = group[offset:offset+args.batch_size]
                    x = tf.stack([r.augment(r.preprocess_legacy(row, r.image_path(row,args.extracted),reference), row["sample_id"], epoch) for row in batch])
                    y = np.array([row["ground_truth_code"] for row in batch],np.int64)
                    loss_sum += accumulator.add(x,y)*len(batch)
                accumulator.flush()  # Flush final 55-sample group as well.
                progress.update(min(logical+100,len(rows)))
            training_seconds = time.perf_counter()-start
            scores = []
            progress = Progress(len(validation),args.log_every,30,r.LOG)
            for offset in range(0,len(validation),args.batch_size):
                batch = validation[offset:offset+args.batch_size]
                x = np.stack([r.preprocess_legacy(row,r.image_path(row,args.extracted),reference) for row in batch])
                scores.extend(model(x,training=False).numpy()[:,1].tolist())
                progress.update(min(offset+args.batch_size,len(validation)))
            history.append(dict(epoch=epoch, boundary=boundary, loss=loss_sum/len(rows), training_seconds=training_seconds,
                                internal_validation=calculate_binary_metrics([row["ground_truth_code"] for row in validation],scores)))
            r.write_json(args.output / "history.json", dict(epochs=history, selection="none; fixed epoch14"))
        del accumulator, optimizer
        gc.collect()
    checkpoint = args.output / "epoch14.keras"
    model.save(checkpoint)
    meta.update(completed_epochs=14, model_sha256=r.sha256_file(checkpoint), completed_at_utc=r.utc_now())
    r.write_json(args.output / "metadata.json",meta)
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-split",action="store_true")
    mode.add_argument("--smoke-benchmark",action="store_true")
    mode.add_argument("--train-production",action="store_true")
    mode.add_argument("--smoke-worker",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--manifest",type=Path,default=Path("manifests/VAL1_manifest.csv"))
    parser.add_argument("--split-output",type=Path,default=Path("manifests/VAL1_internal_v1"))
    parser.add_argument("--extracted",type=Path)
    parser.add_argument("--stain-reference",type=Path,default=r.REFERENCE)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--batch-size",type=int,choices=[1,2,4],default=1)
    parser.add_argument("--log-every",type=int,default=1000)
    parser.add_argument("--log-level",choices=["INFO","DEBUG"],default="INFO")
    parser.add_argument("--smoke-parent",type=Path,help=argparse.SUPPRESS)
    parser.add_argument("--boundary",choices=[s[0] for s in r.SCHEDULE],help=argparse.SUPPRESS)
    parser.add_argument("--precision",choices=["float32","mixed_float16"],default="float32",help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.log_every < 1:
        parser.error("--log-every must be positive")
    if not args.freeze_split and args.output is None:
        parser.error("--output required")
    if (args.smoke_benchmark or args.train_production) and args.extracted is None:
        parser.error("--extracted required")
    if not args.smoke_worker and args.precision != "float32":
        parser.error("Mixed precision only allowed as an explicitly labelled smoke fallback")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.smoke_worker:
        return smoke_worker(args)
    if args.extracted is not None:
        r.ensure_output_separation(args.extracted,[p for p in (args.output,args.split_output) if p is not None])
    train, validation, summary = r.freeze_split(args.manifest,args.split_output)
    if args.freeze_split:
        print(json.dumps(summary,indent=2))
        return 0
    r.logging_setup(args.output,args.log_level)
    meta = r.metadata(summary,"smoke_disposable" if args.smoke_benchmark else "production_pending",args.batch_size)
    if args.smoke_benchmark:
        meta.update(physical_batch=None,physical_batches=[1,2,4])
    r.write_json(args.output / "metadata.json",meta)
    try:
        return smoke_benchmark(args,train,summary) if args.smoke_benchmark else production(args,train,validation,summary)
    except Exception:
        r.LOG.exception("Run failed; no completion claim")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
