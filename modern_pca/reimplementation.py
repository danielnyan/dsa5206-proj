"""Shared contracts for the constrained binary reimplementation (not reproduction).

No TensorFlow import or dataset access at import time. Training accepts only the
authenticated VAL1 manifest; paths are resolved within class-specific VAL1 roots.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import logging
import os
from pathlib import Path
import sys

import numpy as np

from tools.build_validation_manifests import (CLASSES, COUNTS, FIELDS, csv_bytes,
                                            ordered_rows, validate_rows, normalized_path)
from .evaluate_paper import (sha256_file, write_json, utc_now, create_legacy_normalizer,
                             preprocess_legacy, capture_environment, capture_git,
                             preprocessing_settings)

VAL1_SHA = "c483f827985c241106942766062ad58b23c7ad86e6853ade51bb045bb5feff77"
VAL2_SHA = "7b95ac02798e5971fc2722d3c37e8e7613bc1d020dee9a1a5d49ac939cf746d9"
REFERENCE_SHA = "1d31d575c21dcbcf172957b0783c37fcb07872d3129b3a7b3e86ede25ea0a1c1"
REFERENCE = Path("4_WSI_pipeline/WSI_pipeline_v6/images/standard_he_stain_small.jpg")
VALIDATION_COUNTS = {"benign": 18602, "tumour": 10562}
TRAIN_COUNTS = {"benign": 74407, "tumour": 42248}
PARAMETERS = 209812820
SCHEDULE = [("normal_conv_1_17", 7, 1e-5)] + [
    (f"normal_conv_1_{index}", 1, 1e-6) for index in (14, 11, 9, 7, 5, 3, 1)]
DEVIATIONS = [
    "Constrained reimplementation, not exact reproduction or an original checkpoint.",
    "Released binary benign/tumour labels replace gland/nongland/tumour; two outputs.",
    "VAL1 repurposed for training/internal validation; original training data unavailable.",
    "Patch-level internal split: patient/slide identities unavailable; leakage is possible.",
    "All BatchNormalization layers frozen and called in inference mode.",
    "Physical microbatches with sample-weighted gradient accumulation to effective 100.",
    "Modern TensorFlow/Keras/Pillow and StainTools compatibility adapter, no claimed numerical parity.",
]
LOG = logging.getLogger("reimplementation")


def rank_id(sample_id, namespace="VAL1-internal-v1|42|"):
    return hashlib.sha256((namespace + sample_id).encode("utf-8")).hexdigest(), sample_id


def read_manifest(path, cohort="VAL1", expected_sha=None):
    """Check fingerprint before parsing. Does not read any images."""
    payload = Path(path).read_bytes()
    expected_sha = expected_sha or {"VAL1": VAL1_SHA, "VAL2": VAL2_SHA}[cohort]
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError("Manifest SHA-256 mismatch")
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    if reader.fieldnames != FIELDS:
        raise ValueError("Unexpected manifest schema")
    rows = list(reader)
    for row in rows:
        for key in ("row_index", "ground_truth_code", "file_size_bytes", "width", "height"):
            row[key] = int(row[key])
    validate_rows(rows, cohort, COUNTS[cohort])
    return rows


def split_val1(rows, validation_counts=None):
    """Membership by hash rank; each output sorted/reindexed like Stage 2B."""
    validation_counts = VALIDATION_COUNTS if validation_counts is None else validation_counts
    if any(r["cohort"] != "VAL1" or r["class_name"] not in CLASSES or
           r["ground_truth_code"] != CLASSES[r["class_name"]] for r in rows):
        raise ValueError("Training split requires VAL1 and benign=0/tumour=1")
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample_id")
    train, validation = [], []
    for label in CLASSES:
        ranked = sorted((r for r in rows if r["class_name"] == label), key=lambda r: rank_id(r["sample_id"]))
        count = validation_counts[label]
        if not 0 < count < len(ranked):
            raise ValueError("Invalid split class count")
        validation.extend(ranked[:count])
        train.extend(ranked[count:])
    return ordered_rows(train), ordered_rows(validation)


def freeze_split(manifest, output):
    rows = read_manifest(manifest)
    train, validation = split_val1(rows)
    validate_rows(train, "VAL1", TRAIN_COUNTS)
    validate_rows(validation, "VAL1", VALIDATION_COUNTS)
    output = Path(output)
    artifacts = {"VAL1_train.csv": csv_bytes(train), "VAL1_internal_validation.csv": csv_bytes(validation)}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in artifacts.items()}
    # Existing split files must be byte-identical. Never replace a different split.
    for name, data in artifacts.items():
        path = output / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Frozen split differs: {path}")
    output.mkdir(parents=True, exist_ok=True)
    for name, data in artifacts.items():
        if not (output / name).exists():
            (output / name).write_bytes(data)
    summary = dict(source_manifest_sha256=VAL1_SHA, seed=42, algorithm="SHA256 UTF-8 VAL1-internal-v1|42|<sample_id>; ties sample_id; first N per class validate",
                   train=TRAIN_COUNTS, internal_validation=VALIDATION_COUNTS, fingerprints=hashes,
                   row_order="cohort,class_name,normalized relative_path; independent zero-based row_index",
                   limitation=DEVIATIONS[3], source_images_modified=False, VAL2_images_accessed=False)
    sidecar = output / "VAL1_split_summary.json"
    if sidecar.exists():
        import json
        if json.loads(sidecar.read_text()) != summary:
            raise ValueError("Frozen split summary differs")
    else:
        write_json(sidecar, summary)
    return train, validation, summary


def image_path(row, extracted, *, final_evaluation=False):
    cohort, label = row["cohort"], row["class_name"]
    if cohort != "VAL1" and not (cohort == "VAL2" and final_evaluation):
        raise ValueError("VAL2 is forbidden in training/development/benchmarking")
    if label not in CLASSES or row["ground_truth_code"] != CLASSES[label]:
        raise ValueError("Invalid class mapping")
    relative = normalized_path(row["relative_path"])
    if not relative.startswith(label + "/") or row["sample_id"] != cohort + ":" + relative:
        raise ValueError("Invalid image identity/path")
    number = 1 if cohort == "VAL1" else 2
    root = Path(extracted) / f"val_dataset_{number}_{'norm' if label == 'benign' else 'tu'}"
    path = root / relative.split("/", 1)[1]
    # Resolve/check each component to prevent a link redirect into another cohort.
    current = Path(extracted)
    for component in path.relative_to(current).parts:
        current = current / component
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            raise ValueError("Dataset links/junctions are forbidden")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Image escaped its class root")
    return path


def ensure_output_separation(extracted, outputs):
    """Reject writing artifacts anywhere inside (or above) the source dataset."""
    root = Path(extracted).resolve()
    for output in outputs:
        path = Path(output).resolve()
        if path == root or path.is_relative_to(root) or root.is_relative_to(path):
            raise ValueError("Output and source dataset must not overlap")


def normalizers(reference):
    sha256_file(reference, REFERENCE_SHA)
    return create_legacy_normalizer(reference)


def flip_seed(sample_id, epoch):
    digest = hashlib.sha256(f"VAL1-flips-v1|42|{epoch}|{sample_id}".encode()).digest()
    return np.array([int.from_bytes(digest[i:i+4], "big") & 0x7fffffff for i in (0, 4)], dtype=np.int32)


def augment(image, sample_id, epoch):
    import tensorflow as tf
    # One stateless draw per axis, independent of physical batch and traversal order.
    decisions = tf.random.stateless_uniform([2], flip_seed(sample_id, epoch)) < 0.5
    image = tf.cond(decisions[0], lambda: tf.reverse(image, [1]), lambda: image)
    return tf.cond(decisions[1], lambda: tf.reverse(image, [0]), lambda: image)


def configure_stage(model, boundary):
    import tensorflow as tf
    backbone = model.backbone
    names = [layer.name for layer in backbone.layers]
    for expected, _, _ in SCHEDULE:
        if expected not in names:
            raise ValueError(f"Missing NASNet stage boundary: {expected}")
    if boundary not in {s[0] for s in SCHEDULE}:
        raise ValueError("Unknown stage boundary")
    backbone.trainable = True
    boundary_index = names.index(boundary)
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= boundary_index and not isinstance(layer, tf.keras.layers.BatchNormalization)
    return sum(int(np.prod(v.shape)) for v in model.trainable_variables)


def validate_model(model):
    import tensorflow as tf
    if tuple(model.input_shape) != (None, 350, 350, 3) or tuple(model.output_shape) != (None, 2):
        raise ValueError("Require 350x350 RGB input and two outputs")
    tail = model.layers[-3:]
    if not (isinstance(tail[0], tf.keras.layers.Flatten) and
            isinstance(tail[1], tf.keras.layers.Dense) and tail[1].units == 256 and
            tail[1].activation.__name__ == "relu" and
            isinstance(tail[2], tf.keras.layers.Dense) and tail[2].units == 2 and
            tail[2].activation.__name__ == "softmax" and model.count_params() == PARAMETERS):
        raise ValueError("Invalid constrained NASNetLarge head/parameter count")


def build_model(precision="float32"):
    import tensorflow as tf
    tf.keras.mixed_precision.set_global_policy(precision)
    from .models import build_classifier
    model = build_classifier(2, weights="imagenet")
    validate_model(model)
    counts = {boundary: configure_stage(model, boundary) for boundary, _, _ in SCHEDULE}
    return model, counts


def adam(rate):
    import tensorflow as tf
    return tf.keras.optimizers.Adam(learning_rate=rate, beta_1=0.9, beta_2=0.999,
                                    epsilon=1e-7, amsgrad=False)


class Accumulator:
    """Sum sample losses/gradients, divide exactly once by actual group size.

    Compiled microsteps include accumulation and finite checks. A fixed loss
    scale of 128 is used only by the explicitly labelled mixed-float16 fallback;
    gradients are unscaled before float32 accumulation. Nonfinite values abort.
    """
    def __init__(self, model, optimizer, effective=100, loss_scale=1.0):
        import tensorflow as tf
        self.model, self.optimizer, self.effective = model, optimizer, effective
        self.variables = list(model.trainable_variables)
        self.buffers = [tf.Variable(tf.zeros(v.shape, tf.float32), trainable=False) for v in self.variables]
        self.count = 0
        self.scale = float(loss_scale)
        self.micro = tf.function(self._micro, reduce_retracing=True)
        self.apply = tf.function(self._apply, reduce_retracing=True)

    def _micro(self, images, labels):
        import tensorflow as tf
        with tf.GradientTape() as tape:
            probabilities = tf.cast(self.model(images, training=False), tf.float32)
            loss = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(labels, probabilities))
            scaled = loss * self.scale
        gradients = tape.gradient(scaled, self.variables)
        tf.debugging.assert_all_finite(loss, "Nonfinite loss")
        for gradient, buffer in zip(gradients, self.buffers):
            if gradient is None:
                raise ValueError("Disconnected trainable variable")
            gradient = tf.cast(gradient, tf.float32) / self.scale
            tf.debugging.assert_all_finite(gradient, "Nonfinite gradient")
            buffer.assign_add(gradient)
        return loss

    def _apply(self, count):
        import tensorflow as tf
        self.optimizer.apply_gradients([(b / tf.cast(count, tf.float32), v)
                                        for b, v in zip(self.buffers, self.variables)])
        for buffer in self.buffers:
            buffer.assign(tf.zeros_like(buffer))
        return self.optimizer.iterations

    def add(self, images, labels):
        size = int(images.shape[0])
        if size <= 0 or self.count + size > self.effective:
            raise ValueError("Microbatch crosses a logical batch boundary")
        loss = float(self.micro(images, labels).numpy())
        self.count += size
        return loss / size

    def flush(self):
        import tensorflow as tf
        if self.count:
            self.apply(tf.constant(self.count, tf.int32)).numpy()
            self.count = 0


def weight_hashes(variables):
    return [hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in variables]


def gpu_setup(precision="float32"):
    import tensorflow as tf
    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError("GPU smoke/production requires a TensorFlow-visible GPU; no CPU fallback")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.experimental.set_synchronous_execution(True)
    tf.config.experimental.enable_op_determinism()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.keras.utils.set_random_seed(42)
    tf.keras.mixed_precision.set_global_policy(precision)
    return dict(tensorflow=tf.__version__, keras=getattr(tf.keras, "__version__", None),
                visible_gpus=[str(d) for d in devices],
                gpu_details=[tf.config.experimental.get_device_details(d) for d in devices],
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
                build=tf.sysconfig.get_build_info(), synchronous_execution=True,
                tensor_float_32=False, deterministic_ops=True)


def metadata(split_summary, purpose, physical_batch, precision="float32"):
    cache = Path(os.environ.get("KERAS_HOME", Path.home() / ".keras")) / "models"
    weights = {p.name: sha256_file(p) for p in cache.glob("nasnet_large_no_top.h5")} if cache.exists() else {}
    return dict(created_at_utc=utc_now(), purpose=purpose, code=capture_git(),
                implementation_sha256={p.name: sha256_file(p) for p in [
                    *Path(__file__).parent.glob("*reimplementation*.py"),
                    Path(__file__).with_name("models.py"), Path(__file__).with_name("evaluate_paper.py"),
                    Path(__file__).resolve().parents[1] / "tools/build_validation_manifests.py"]},
                split=split_summary, seed=42, class_order=list(CLASSES),
                model=dict(backbone="NASNetLarge", weights="imagenet", input=[350,350,3],
                           head=["Flatten", "Dense(256,relu)", "Dense(2,softmax)"], parameters=PARAMETERS),
                preprocessing=dict(preprocessing_settings(), resize="Pillow LANCZOS 350x350", dtype="float32", scale="1/255"), stain_reference_sha256=REFERENCE_SHA,
                imagenet_weight_files=weights, environment=capture_environment(),
                physical_batch=physical_batch, effective_batch=100, precision=precision,
                optimizer=dict(name="Adam", beta_1=0.9, beta_2=0.999, epsilon=1e-7,
                               amsgrad=False, weight_decay=None, clipping=None, reset="each stage; retain epochs 1-7"),
                schedule=SCHEDULE, deviations=DEVIATIONS, command=sys.argv)


def logging_setup(output, level="INFO"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger()
    logger.setLevel(level)
    logging.captureWarnings(True)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(output / "run.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    LOG.info("Run started UTC=%s command=%s", utc_now(), sys.argv)
    return LOG
