"""Fresh standalone Model 2 or native-versus-ENS evaluation on VAL2.

This is inference only. ENS mode invokes Model 2 only for native scores in the
inclusive [0.2, 0.8] grey zone; Model-2-only mode scores every selected patch.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from . import ens_cache as e
from .ens_train import validate_model as validate_model2
from modern_pca import reimplementation as r
from modern_pca.evaluate_paper import calculate_binary_metrics, csv_bytes

NATIVE_SHA256 = "433dd48aae800c659d1cba092af2859499bb804d2228ff978a6e9c9602b2adde"
MODEL2_SHA256 = "49305a192b3dfbdad7c9cea641f4195ae096389df0655863022c01f0d58a4680"
VAL2_ARCHIVE_MD5 = {
    "benign": "029069ab405a76d42fcdefac32982989",
    "tumour": "d72b78d364fbfed95d8f0d71d8f2d01e",
}
LOW, HIGH = 0.2, 0.8


class Val2ZipSource:
    """Read VAL2 images only after archive and per-image checksums match."""

    def __init__(self, paths, expected_md5=None):
        self.paths = {label: Path(path) for label, path in paths.items()}
        self.expected_md5 = VAL2_ARCHIVE_MD5 if expected_md5 is None else expected_md5
        self.archives = {}
        for label, path in self.paths.items():
            if label not in r.CLASSES:
                raise ValueError(f"Unknown class archive: {label}")
            if e.md5_file(path) != self.expected_md5[label]:
                raise ValueError(f"VAL2 archive MD5 mismatch: {path}")
            self.archives[label] = zipfile.ZipFile(path)

    @staticmethod
    def member(row):
        label = row["class_name"]
        if row["cohort"] != "VAL2":
            raise ValueError("VAL2 inference accepts VAL2 rows only")
        if label not in r.CLASSES or row["ground_truth_code"] != r.CLASSES[label]:
            raise ValueError("Invalid VAL2 class mapping")
        relative = r.normalized_path(row["relative_path"])
        if not relative.startswith(label + "/") or row["sample_id"] != "VAL2:" + relative:
            raise ValueError("Invalid VAL2 image identity/path")
        return relative.split("/", 1)[1]

    def check(self, rows):
        names = {label: set(archive.namelist()) for label, archive in self.archives.items()}
        missing = [row["sample_id"] for row in rows
                   if self.member(row) not in names[row["class_name"]]]
        if missing:
            raise ValueError(f"{len(missing)} VAL2 manifest images absent from ZIPs: {missing[:5]}")

    def read(self, row):
        payload = self.archives[row["class_name"]].read(self.member(row))
        if hashlib.sha256(payload).hexdigest() != row["file_sha256"]:
            raise ValueError(f"VAL2 source image SHA-256 mismatch: {row['sample_id']}")
        return payload

    def close(self):
        for archive in self.archives.values():
            archive.close()


def route_ens_scores(native_scores, model2_scores, low=LOW, high=HIGH):
    """Replace native scores inside the uncertainty band; never average/fuse them."""
    native = np.asarray(native_scores, dtype=np.float64)
    second = np.asarray(model2_scores, dtype=np.float64)
    if native.ndim != 1 or native.shape != second.shape or not len(native):
        raise ValueError("Require nonempty, aligned one-dimensional scores")
    both = np.concatenate((native, second))
    if not np.isfinite(both).all() or (both < 0).any() or (both > 1).any():
        raise ValueError("Tumour probabilities must be finite and in [0, 1]")
    triggered = (native >= low) & (native <= high)
    return np.where(triggered, second, native), triggered


def _validate_checkpoints(native_path, model2_path):
    native_path, model2_path = Path(native_path), Path(model2_path)
    if r.sha256_file(native_path) != NATIVE_SHA256:
        raise ValueError("Native checkpoint SHA-256 does not match the release")
    _validate_model2_checkpoint(model2_path)
    return native_path, model2_path


def _validate_model2_checkpoint(model2_path):
    model2_path = Path(model2_path)
    if r.sha256_file(model2_path) != MODEL2_SHA256:
        raise ValueError("Model 2 checkpoint SHA-256 does not match its release metadata")
    return model2_path


def _probabilities(model, batch):
    values = np.asarray(model(batch, training=False).numpy(), dtype=np.float64)
    if (values.shape != (len(batch), 2) or not np.isfinite(values).all() or
            np.any(values < 0) or np.any(values > 1) or
            np.any(np.abs(values.sum(axis=1) - 1) > 1e-5)):
        raise ValueError("Invalid two-class softmax output")
    return values[:, 1]


def _predict_model1(model, rows, source, normalizers, batch_size):
    from modern_pca.evaluate_reimplementation import probabilities_2
    scores = []
    for start in range(0, len(rows), batch_size):
        group = rows[start:start + batch_size]
        images = np.stack([
            r.preprocess_legacy(row, e.Payload(source.read(row)), normalizers)
            for row in group
        ])
        scores.extend(_probabilities(model, images).tolist())
        completed = start + len(group)
        if completed == len(rows) or completed % 1000 < batch_size:
            print(f"Native inference: {completed}/{len(rows)}", flush=True)
    return probabilities_2(np.column_stack((1 - np.asarray(scores), scores)), len(rows))[:, 1]


def _predict_model2(model, rows, source, normalizers, batch_size):
    scores = []
    for start in range(0, len(rows), batch_size):
        group = rows[start:start + batch_size]
        images = np.stack([
            e.to_float(e.model2_pixels(row, source.read(row), normalizers))
            for row in group
        ])
        scores.extend(_probabilities(model, images).tolist())
        completed = start + len(group)
        if completed == len(rows) or completed % 1000 < batch_size:
            print(f"Model 2 inference: {completed}/{len(rows)}", flush=True)
    return np.asarray(scores, dtype=np.float64)


def run(args):
    import tensorflow as tf

    started = time.perf_counter()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    if args.model2_only:
        model2_path = _validate_model2_checkpoint(args.model2_model)
        native_path = None
    else:
        native_path, model2_path = _validate_checkpoints(args.native_model, args.model2_model)
    rows = r.read_manifest(args.manifest, "VAL2", expected_sha=r.VAL2_SHA)
    if args.max_per_class is not None:
        rows = [row for label in r.CLASSES
                for row in [item for item in rows if item["class_name"] == label][:args.max_per_class]]
    if not rows:
        raise ValueError("No VAL2 samples selected")
    source = Val2ZipSource({
        "benign": args.archives / "val_dataset_2_norm.zip",
        "tumour": args.archives / "val_dataset_2_tu.zip",
    })
    try:
        source.check(rows)
        reference = r.normalizers(args.stain_reference)
        model2_model = tf.keras.models.load_model(model2_path, compile=False)
        validate_model2(model2_model)
        model2_model.trainable = False

        if args.model2_only:
            model2_scores = _predict_model2(model2_model, rows, source, reference, args.batch_size)
            triggered = np.ones(len(rows), dtype=bool)
            final_scores = model2_scores
        else:
            native_model = tf.keras.models.load_model(native_path, compile=False)
            r.validate_model(native_model)
            native_model.trainable = False
            native_scores = _predict_model1(native_model, rows, source, reference, args.batch_size)
            triggered = (native_scores >= LOW) & (native_scores <= HIGH)
            trigger_indices = np.flatnonzero(triggered)
            model2_scores = np.full(len(rows), np.nan, dtype=np.float64)
            if len(trigger_indices):
                selected_rows = [rows[int(index)] for index in trigger_indices]
                model2_scores[trigger_indices] = _predict_model2(
                    model2_model, selected_rows, source, reference, args.batch_size)
            final_scores, routed = route_ens_scores(native_scores, np.nan_to_num(model2_scores))
            if not np.array_equal(triggered, routed):
                raise AssertionError("ENS trigger mask changed during routing")

        labels = [row["ground_truth_code"] for row in rows]
        predictions = []
        if args.model2_only:
            for index, row in enumerate(rows):
                predictions.append(dict(
                    row_index=row["row_index"], sample_id=row["sample_id"],
                    ground_truth_code=row["ground_truth_code"],
                    p_tumour_model2=float(model2_scores[index]),
                    model2_prediction=int(model2_scores[index] > 0.5),
                ))
        else:
            for index, row in enumerate(rows):
                predictions.append(dict(
                    row_index=row["row_index"], sample_id=row["sample_id"],
                    ground_truth_code=row["ground_truth_code"],
                    p_tumour_native=float(native_scores[index]),
                    native_prediction=int(native_scores[index] > 0.5),
                    p_tumour_model2=(float(model2_scores[index]) if triggered[index] else ""),
                    model2_invoked=int(triggered[index]),
                    p_tumour_ens=float(final_scores[index]),
                    ens_prediction=int(final_scores[index] > 0.5),
                ))

        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        mode = "model2" if args.model2_only else "ens"
        prediction_path = output / f"val2_{mode}_predictions.csv"
        prediction_path.write_bytes(csv_bytes(predictions, list(predictions[0])))
        metrics = dict(
            cohort="VAL2", samples=len(rows),
            evaluation_mode="model2_only" if args.model2_only else "ens",
            full_cohort=len(rows) == 34103,
            max_per_class=args.max_per_class,
            class_counts=dict(Counter(row["class_name"] for row in rows)),
            rule=("Use Model 2 for every patch; tumour iff p > 0.5" if args.model2_only else
                "Use native outside [0.2, 0.8]; within the inclusive band use Model 2; tumour iff p > 0.5"),
            model2_invoked=int(triggered.sum()), trigger_fraction=float(triggered.mean()),
            model2=calculate_binary_metrics(labels, model2_scores) if args.model2_only else None,
            native=None if args.model2_only else calculate_binary_metrics(labels, native_scores),
            ens=None if args.model2_only else calculate_binary_metrics(labels, final_scores),
            native_model_sha256=None if args.model2_only else NATIVE_SHA256,
            model2_sha256=MODEL2_SHA256,
            manifest_sha256=r.sha256_file(args.manifest),
            archive_md5={label: e.md5_file(path) for label, path in source.paths.items()},
            stain_reference_sha256=r.sha256_file(args.stain_reference),
            tensorflow=tf.__version__, keras=tf.keras.__version__,
            elapsed_seconds=time.perf_counter() - started,
            evaluation_scope="VAL2 held out from VAL1 training; constrained reimplementation, not exact paper reproduction",
        )
        metrics_path = output / f"val2_{mode}_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        completion = dict(status="PASS", completed_at_utc=r.utc_now(), samples=len(rows),
                          artifacts={p.name: r.sha256_file(p) for p in (prediction_path, metrics_path)})
        (output / "run_complete.json").write_text(
            json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
        return metrics
    finally:
        source.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-model", type=Path)
    parser.add_argument("--model2-model", type=Path, required=True)
    parser.add_argument("--model2-only", action="store_true",
                        help="Evaluate Model 2 directly on every VAL2 patch, without native routing")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--stain-reference", type=Path, default=r.REFERENCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Deterministic smoke subset size per class; omit for full VAL2")
    args = parser.parse_args(argv)
    if args.max_per_class is not None and args.max_per_class < 1:
        parser.error("--max-per-class must be positive")
    if not args.model2_only and args.native_model is None:
        parser.error("--native-model is required unless --model2-only is set")
    run(args)


if __name__ == "__main__":
    main()