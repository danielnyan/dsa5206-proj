"""Strict native C1 evaluation of an existing Tolkach tumour/benign model.

No model construction, training, splitting, TTA, or network access is performed.
Legacy evidence: 1_training/NASNetLarge_350px_FULL_training.py:9-16;
2_validation_Tumor_vs_Benign/Validation_Tu_vs_N_native(C1)_and_C8.py:38-44,
101-123; 4_WSI_pipeline/WSI_pipeline_v6/wsi_maps.py:23-27. The latter supplies
the strict >0.5 tie policy; the patch script has label-dependent tie handling.

Runtime requirements: the repository's NumPy/Pillow/scikit-learn/TensorFlow
dependencies, PLUS a working original staintools installation and its native
dependencies (historical upstream reference: staintools 2.1.2). Nothing is
installed automatically. Numerical parity with the historical environment must
be established separately. Pillow Resampling.LANCZOS replaces Image.ANTIALIAS;
this preserves the filter, not a claim of cross-version pixel identity.

Paper checkpoint sidecar JSON requires: model_sha256, source, acquisition_date,
identity ("native"), architecture ("NASNetLarge"), class_order
(["gland", "nongland", "tumour"]), class_order_evidence, conversion_history
(list), input_shape ([null,350,350,3]), output_shape ([null,3]). A valid sidecar
is a provenance declaration, not independent authentication of authorship.

One pass, zero warm-up only. Total timing ends before final JSON timing reports
and completion logging; that self-reporting tail is explicitly excluded.
Only a validated run_complete.json marks success; per-file statuses are provisional.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import inspect
import json
import logging
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import unicodedata
import uuid
import zipfile
from types import FunctionType, SimpleNamespace
from typing import Any, Callable

import numpy as np


CLASS_ORDER = ["gland", "nongland", "tumour"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
AMBIGUOUS_SUFFIXES = {".gif", ".webp", ".ico", ".svs", ".ndpi", ".heic", ".avif"}
MANIFEST_FIELDS = ["row_index", "sample_id", "cohort", "source_group", "relative_path",
                   "ground_truth", "ground_truth_code", "file_size_bytes", "file_sha256",
                   "source_width", "source_height", "source_mode"]
PREDICTION_FIELDS = ["row_index", "sample_id", "cohort", "relative_path", "ground_truth",
                     "ground_truth_code", "p_gland", "p_nongland", "p_benign", "p_tumour",
                     "predicted_binary_label", "predicted_binary_code", "correct"]
TIMING_PHASES = ["startup", "model_load", "manifest", "reference_setup", "warmup",
                 "preprocessing", "inference", "metrics", "output", "integrity"]
TIMING_SCOPE = {"inference_includes_transfer": True, "inference_output_materialized": True,
                "total_includes_warmup": True, "warmup_policy": "zero calls",
                "total_excludes_final_timing_report_write": True,
                "total_excludes_completion_log": True,
                "total_excludes_completion_marker": True,
                "output_excludes_final_timing_reports": True}


class EvaluationError(RuntimeError):
    """A scientific contract, artifact integrity, or runtime failure."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, expected: str | None = None) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    result = digest.hexdigest()
    if expected is not None and result != expected.lower():
        raise EvaluationError(f"SHA-256 mismatch: {path}; expected {expected}, got {result}")
    return result


def fingerprint(config: dict) -> str:
    """Hash only the explicitly selected scientific configuration, not runtime settings."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def csv_bytes(rows: list[dict], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: format(value, ".17g") if isinstance(value, (float, np.floating))
                         else value for key, value in row.items() if key in fields})
    return stream.getvalue().encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_json(path: Path, document: dict) -> None:
    atomic_write(path, (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


@dataclass
class Timings:
    clock: Callable[[], float] = time.perf_counter
    seconds: dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in TIMING_PHASES})

    @contextmanager
    def measure(self, phase: str):
        started = self.clock()
        try:
            yield
        finally:
            self.seconds[phase] += self.clock() - started

    def report(self, count: int, total: float) -> dict:
        result = {f"{name}_seconds": value for name, value in self.seconds.items()}
        result.update(total_seconds=total,
                      average_preprocessing_ms_per_image=1000 * self.seconds["preprocessing"] / count if count else None,
                      average_inference_ms_per_image=1000 * self.seconds["inference"] / count if count else None,
                      end_to_end_ms_per_image=1000 * total / count if count else None,
                      inference_images_per_second=count / self.seconds["inference"] if count and self.seconds["inference"] else None,
                      end_to_end_images_per_second=count / total if count and total else None)
        return result


@dataclass
class Manifest:
    rows: list[dict]
    paths: dict[str, Path]
    roots: dict[str, Path]
    duplicate_content: list[list[str]]
    ignored_files: list[str]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(csv_bytes(self.rows, MANIFEST_FIELDS)).hexdigest()

    @property
    def counts(self) -> dict:
        benign = sum(row["ground_truth_code"] == 0 for row in self.rows)
        return {"total": len(self.rows), "benign": benign, "tumour": len(self.rows) - benign}


def check_logical_paths(paths: list[str]) -> None:
    seen: set[str] = set()
    for path in paths:
        key = unicodedata.normalize("NFC", path).casefold()
        if key in seen:
            raise EvaluationError(f"Duplicate/case/Unicode logical path collision: {path}")
        seen.add(key)


def _raise_walk_error(error: OSError) -> None:
    raise EvaluationError(f"Cannot enumerate dataset: {error}") from error


def build_manifest(benign_dir: Path, tumour_dir: Path, cohort: str) -> Manifest:
    """Inventory every image under explicit label roots without any split or sampling."""
    from PIL import Image, UnidentifiedImageError

    roots = {"benign": Path(benign_dir).resolve(strict=True), "tumour": Path(tumour_dir).resolve(strict=True)}
    a, b = roots.values()
    if a == b or a in b.parents or b in a.parents:
        raise EvaluationError("Benign and tumour roots must be disjoint directories")
    rows, ignored = [], []
    paths: dict[str, Path] = {}
    resolved_seen: set[Path] = set()
    inode_seen: set[tuple[int, int]] = set()
    by_content: dict[str, list[dict]] = {}
    for label, (group, root) in enumerate(roots.items()):
        if not root.is_dir():
            raise EvaluationError(f"Not a directory: {root}")
        group_count = 0
        for parent, dirs, files in os.walk(root, followlinks=False, onerror=_raise_walk_error):
            for directory in dirs:
                if (Path(parent) / directory).is_symlink():
                    raise EvaluationError(f"Directory symlink is ambiguous: {Path(parent) / directory}")
            for filename in files:
                path = Path(parent) / filename
                logical = group + "/" + path.relative_to(root).as_posix()
                if path.suffix.lower() in AMBIGUOUS_SUFFIXES:
                    raise EvaluationError(f"Unsupported/ambiguous image format: {logical}")
                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    try:
                        with Image.open(path):
                            pass
                    except UnidentifiedImageError:
                        ignored.append(logical)
                        continue
                    raise EvaluationError(f"Image has unsupported extension: {logical}")
                resolved = path.resolve(strict=True)
                stat = path.stat()
                inode = (stat.st_dev, stat.st_ino)
                if resolved in resolved_seen or (stat.st_ino and inode in inode_seen):
                    raise EvaluationError(f"Duplicate resolved physical path: {logical}")
                resolved_seen.add(resolved)
                if stat.st_ino:
                    inode_seen.add(inode)
                payload = path.read_bytes()
                try:
                    with Image.open(io.BytesIO(payload)) as image:
                        if getattr(image, "n_frames", 1) != 1:
                            raise EvaluationError("Multi-frame image is ambiguous")
                        width, height, mode = image.width, image.height, image.mode
                        image.load()
                except Exception as error:
                    raise EvaluationError(f"Unreadable/ambiguous image {logical}: {error}") from error
                digest = hashlib.sha256(payload).hexdigest()
                row = dict(row_index=0, sample_id=f"{cohort}:{logical}", cohort=cohort,
                           source_group=group, relative_path=logical, ground_truth=group,
                           ground_truth_code=label, file_size_bytes=len(payload), file_sha256=digest,
                           source_width=width, source_height=height, source_mode=mode)
                if any(other["ground_truth_code"] != label for other in by_content.get(digest, [])):
                    raise EvaluationError(f"Conflicting-label duplicate content: {logical}")
                by_content.setdefault(digest, []).append(row)
                rows.append(row)
                paths[row["sample_id"]] = resolved
                group_count += 1
        if not group_count:
            raise EvaluationError(f"Empty {group} image directory: {root}")
    check_logical_paths([row["relative_path"] for row in rows])
    rows.sort(key=lambda row: row["relative_path"])
    for index, row in enumerate(rows):
        row["row_index"] = index
    duplicates = sorted(sorted(row["relative_path"] for row in group) for group in by_content.values() if len(group) > 1)
    return Manifest(rows, paths, roots, duplicates, sorted(ignored))


def validate_manifest(manifest: Manifest) -> None:
    """Reinventory and rehash at completion; added, removed or changed images fail."""
    current = build_manifest(manifest.roots["benign"], manifest.roots["tumour"], manifest.rows[0]["cohort"])
    if current.sha256 != manifest.sha256 or current.ignored_files != manifest.ignored_files:
        raise EvaluationError("Dataset manifest changed during evaluation")
    if any(current.paths[key] != path for key, path in manifest.paths.items()):
        raise EvaluationError("Dataset physical paths changed during evaluation")


def load_checkpoint_metadata(path: Path | None, model_sha256: str, purpose: str,
                             class_order: list[str] | None) -> dict:
    if path is None:
        if purpose == "paper":
            raise EvaluationError("Paper mode requires --checkpoint-metadata")
        metadata = {"class_order": class_order, "provenance_status": "unverified sanity fixture"}
    else:
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as error:
            raise EvaluationError(f"Checkpoint metadata must be valid UTF-8 JSON: {error}") from error
        if not isinstance(metadata, dict):
            raise EvaluationError("Checkpoint metadata JSON top level must be an object")
        checksum = metadata.get("model_sha256")
        if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-fA-F]{64}", checksum) is None:
            raise EvaluationError("Checkpoint metadata model_sha256 must be a 64-character SHA-256 string")
        if checksum.lower() != model_sha256:
            raise EvaluationError("Checkpoint metadata model SHA-256 mismatch")
    if not isinstance(metadata.get("class_order"), list):
        raise EvaluationError("Checkpoint class_order must be a list; unknown class mapping")
    if metadata.get("class_order") != CLASS_ORDER or (class_order is not None and class_order != CLASS_ORDER):
        raise EvaluationError("Unknown/conflicting class mapping; require gland,nongland,tumour")
    if purpose == "paper":
        required = ["source", "acquisition_date", "class_order_evidence", "conversion_history",
                    "architecture", "identity", "input_shape", "output_shape"]
        if any(key not in metadata for key in required):
            raise EvaluationError(f"Incomplete checkpoint provenance; required fields: {required}")
        for key in ["input_shape", "output_shape"]:
            shape = metadata[key]
            if (not isinstance(shape, list) or not shape or shape[0] is not None
                    or any(type(dim) is not int or dim <= 0 for dim in shape[1:])):
                raise EvaluationError(f"Checkpoint {key} must be a list [null, positive integer dimensions]")
        if any(not isinstance(metadata[key], str) or not metadata[key].strip()
               for key in ["source", "acquisition_date", "class_order_evidence"]):
            raise EvaluationError("Checkpoint source, acquisition date and class-order evidence must be nonempty")
        if (metadata["architecture"] != "NASNetLarge" or metadata["identity"] != "native"
                or metadata["input_shape"] != [None, 350, 350, 3]
                or metadata["output_shape"] != [None, 3]
                or not isinstance(metadata["conversion_history"], list)):
            raise EvaluationError("Checkpoint provenance conflicts with native NASNetLarge C1 contract")
    return metadata


def _activation_name(layer: Any) -> str:
    return getattr(getattr(layer, "activation", None), "__name__", "")


# Inference-relevant configuration for every operation in the NASNet-A graph.
# Initializers/regularizers/trainable flags do not affect inference with loaded weights.
GRAPH_DEFAULTS = {
    "InputLayer": {},
    "Conv2D": dict(strides=(1, 1), padding="valid", data_format="channels_last",
                   dilation_rate=(1, 1), groups=1, activation="linear", use_bias=True),
    "SeparableConv2D": dict(strides=(1, 1), padding="valid", data_format="channels_last",
                            dilation_rate=(1, 1), depth_multiplier=1, activation="linear", use_bias=True),
    "BatchNormalization": dict(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True),
    "Activation": {}, "ZeroPadding2D": dict(data_format="channels_last"),
    "Cropping2D": dict(data_format="channels_last"),
    "AveragePooling2D": dict(padding="valid", data_format="channels_last"),
    "MaxPooling2D": dict(padding="valid", data_format="channels_last"),
    "Add": {}, "Concatenate": dict(axis=-1),
    "Flatten": dict(data_format="channels_last"),
    "Dense": dict(activation="linear", use_bias=True),
}
GRAPH_REQUIRED = {"Conv2D": ("filters", "kernel_size"), "SeparableConv2D": ("filters", "kernel_size"),
                  "Activation": ("activation",), "ZeroPadding2D": ("padding",),
                  "Cropping2D": ("cropping",), "AveragePooling2D": ("pool_size", "strides"),
                  "MaxPooling2D": ("pool_size", "strides"), "Dense": ("units",)}


def operation_config(kind: str, config: dict) -> dict:
    if kind not in GRAPH_DEFAULTS:
        raise EvaluationError(f"Unsupported NASNetLarge operation: {kind}")
    result = {key: config.get(key, default) for key, default in GRAPH_DEFAULTS[kind].items()}
    for key in GRAPH_REQUIRED.get(kind, ()):
        if key not in config:
            raise EvaluationError(f"Missing NASNetLarge configuration {kind}.{key}")
        result[key] = config[key]
    if isinstance(result.get("axis"), (list, tuple)) and len(result["axis"]) == 1:
        result["axis"] = result["axis"][0]  # Keras 2 BN serialized a singleton axis list.
    if result.get("axis") == 3:
        result["axis"] = -1
    return json.loads(json.dumps(result))


class GraphTensor:
    """Shape/configuration record only: no tensors, variables, RNG or inference."""
    def __init__(self, kind, config, parents, shape, parameters=0):
        self.kind, self.config, self.parents = kind, config, parents
        self.shape, self.parameters = tuple(shape), parameters


class GraphLayers:
    """Execute Keras' NASNet topology builder using weight-free shape records."""
    def Input(self, shape, **kwargs):
        return GraphTensor("InputLayer", {}, [], (None, *shape))

    def add(self, inputs, **kwargs):
        return self.Add(**kwargs)(inputs)

    def concatenate(self, inputs, **kwargs):
        return self.Concatenate(**kwargs)(inputs)

    def __getattr__(self, kind):
        if kind not in GRAPH_DEFAULTS:
            raise EvaluationError(f"Unsupported reference operation: {kind}")
        def factory(*args, **kwargs):
            positional = GRAPH_REQUIRED.get(kind, ())
            raw = {**dict(zip(positional, args)), **kwargs}
            if kind.endswith("Pooling2D"):
                raw.setdefault("strides", raw["pool_size"])
            cfg = operation_config(kind, raw)
            def connect(inputs):
                parents = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
                shape = list(parents[0].shape)
                params = 0
                if kind in ("Conv2D", "SeparableConv2D"):
                    channels, filters = shape[-1], cfg["filters"]
                    kh, kw = cfg["kernel_size"]
                    if kind == "Conv2D":
                        params = kh * kw * channels * filters
                    else:
                        depth = cfg["depth_multiplier"]
                        params = kh * kw * channels * depth + channels * depth * filters
                    params += filters if cfg["use_bias"] else 0
                    shape[-1] = filters
                if kind in ("Conv2D", "SeparableConv2D", "AveragePooling2D", "MaxPooling2D"):
                    kernel = cfg.get("kernel_size", cfg.get("pool_size"))
                    for axis in (1, 2):
                        size, stride = shape[axis], cfg["strides"][axis - 1]
                        shape[axis] = ((size + stride - 1) // stride if cfg["padding"] == "same"
                                       else (size - kernel[axis - 1]) // stride + 1)
                elif kind in ("ZeroPadding2D", "Cropping2D"):
                    amounts = cfg["padding" if kind == "ZeroPadding2D" else "cropping"]
                    for axis in (1, 2):
                        shape[axis] += sum(amounts[axis - 1]) * (1 if kind == "ZeroPadding2D" else -1)
                elif kind == "Concatenate":
                    shape[cfg["axis"]] = sum(parent.shape[cfg["axis"]] for parent in parents)
                elif kind == "BatchNormalization":
                    params = shape[cfg["axis"]] * (2 + int(cfg["center"]) + int(cfg["scale"]))
                elif kind == "Flatten":
                    shape = [None, int(np.prod(shape[1:]))]
                elif kind == "Dense":
                    params = shape[-1] * cfg["units"] + (cfg["units"] if cfg["use_bias"] else 0)
                    shape[-1] = cfg["units"]
                return GraphTensor(kind, cfg, parents, shape, params)
            return connect
        return factory


def graph_document(output: GraphTensor) -> dict:
    nodes, seen = [], {}
    def visit(tensor):
        if id(tensor) not in seen:
            parents = [visit(parent) for parent in tensor.parents]
            seen[id(tensor)] = len(nodes)
            nodes.append(dict(operation=tensor.kind, config=tensor.config, inputs=parents,
                              shape=list(tensor.shape), parameters=tensor.parameters))
        return seen[id(tensor)]
    root = visit(output)
    return dict(nodes=nodes, output=root, parameters=sum(node["parameters"] for node in nodes))


def _nasnet_namespace() -> dict:
    import tensorflow as tf
    return inspect.unwrap(tf.keras.applications.NASNetLarge).__globals__


def reference_architecture() -> dict:
    """Derive NASNetLarge from installed Keras source, without building a model.

    Legacy training lines 9–16 specify include_top=False, 350 RGB, Flatten,
    Dense256/ReLU and Dense3/softmax. Keras NASNetLarge specifies 4032 filters,
    6 blocks, 96 stem filters, skip_reduction=True, multiplier=2. Execute those
    SAME builder functions with GraphLayers, never Keras layers/weights.
    Unsupported builder/API changes fail closed and require compatibility review.
    Parameter counts include BN moving statistics, independent of trainable flags.
    """
    try:
        source = _nasnet_namespace()
        names = ["NASNet", "NASNetLarge", "_separable_conv_block", "_adjust_block",
                 "_normal_a_cell", "_reduction_a_cell"]
        def correct_pad(tensor, kernel):
            kernel = (kernel, kernel) if isinstance(kernel, int) else kernel
            return tuple((k // 2 - (1 - tensor.shape[i + 1] % 2), k // 2) for i, k in enumerate(kernel))
        namespace = dict(layers=GraphLayers(),
                         backend=SimpleNamespace(image_data_format=lambda: "channels_last",
                                                 name_scope=lambda *args: nullcontext()),
                         imagenet_utils=SimpleNamespace(obtain_input_shape=lambda shape, **kw: shape,
                                                       correct_pad=correct_pad),
                         Functional=lambda inputs, output, **kwargs: output)
        for name in names:
            function = inspect.unwrap(source[name])
            namespace[name] = FunctionType(function.__code__, namespace, name, function.__defaults__)
        output = namespace["NASNetLarge"](input_shape=(350, 350, 3), include_top=False, weights=None)
        backbone = graph_document(output)
        layers = namespace["layers"]
        output = layers.Dense(3, activation="softmax")(
            layers.Dense(256, activation="relu")(layers.Flatten()(output)))
        full = graph_document(output)
        # Derived: backbone 84,916,818; head (11*11*4032+1)*256 + (256+1)*3.
        if backbone["parameters"] != 84916818 or full["parameters"] != 209813077:
            raise EvaluationError("Installed NASNetLarge definition changed; architecture compatibility review required")
        return {"backbone": backbone, "full": full}
    except Exception as error:
        raise EvaluationError(f"Cannot derive weight-free NASNetLarge contract: {error}") from error


def loaded_architecture(model: Any) -> dict:
    """Traverse actual tensor connectivity; allow only standard Keras operations.

    Nested backbone is expanded, binding its internal input to the outer input.
    Layer names, generated IDs and trainability do not enter the fingerprint.
    Shared/reused layers are rejected for this non-weight-sharing architecture.
    """
    import tensorflow as tf
    memo, layers_seen = {}, set()
    def visit(tensor, bindings):
        if id(tensor) in bindings:
            return bindings[id(tensor)]
        if id(tensor) in memo:
            return memo[id(tensor)]
        history = tensor._keras_history
        layer = history.operation if hasattr(history, "operation") else history[0]
        node_index, tensor_index = history[1], history[2]
        if tensor_index != 0 or id(layer) in layers_seen:
            raise EvaluationError("Unsupported reused/multi-output NASNetLarge operation")
        layers_seen.add(id(layer))
        kind = type(layer).__name__
        node = layer._inbound_nodes[node_index]
        arguments = getattr(node, "arguments", None)
        kwargs = getattr(arguments, "kwargs", getattr(node, "call_kwargs", {}))
        # A forced training=True BN call changes inference despite identical weights
        # and graph edges. Only inference-equivalent optional call flags are allowed.
        if any(not (key == "training" and (value is None or value is False))
               and not (key == "mask" and value is None) for key, value in kwargs.items()):
            raise EvaluationError("Nonstandard NASNetLarge call arguments")
        incoming = node.input_tensors
        incoming = list(incoming) if isinstance(incoming, (list, tuple)) else [incoming]
        if kind == "InputLayer":
            parents = []
        else:
            parents = [visit(item, bindings) for item in incoming]
        if isinstance(layer, tf.keras.Model):
            if len(layer.inputs) != 1 or len(layer.outputs) != 1 or len(parents) != 1:
                raise EvaluationError("Unsupported nested NASNetLarge graph")
            result = visit(layer.outputs[0], {**bindings, id(layer.inputs[0]): parents[0]})
        else:
            if kind not in GRAPH_DEFAULTS or type(layer) is not getattr(tf.keras.layers, kind, None):
                raise EvaluationError(f"Nonstandard NASNetLarge operation: {kind}")
            result = GraphTensor(kind, operation_config(kind, layer.get_config()), parents,
                                 tuple(tensor.shape), layer.count_params())
        memo[id(tensor)] = result
        return result
    try:
        return graph_document(visit(model.outputs[0], {}))
    except EvaluationError:
        raise
    except Exception as error:
        raise EvaluationError(f"Unsupported NASNetLarge graph serialization: {error}") from error


def validate_architecture_document(actual: dict, expected: dict) -> str:
    if actual != expected:
        raise EvaluationError("NASNetLarge operation/configuration/connectivity contract mismatch")
    return fingerprint(actual)


def validate_model_contract(model: Any, metadata: dict, purpose: str = "paper") -> dict:
    """Validate shapes AND the released topology; never infer class semantics from shape."""
    if metadata.get("class_order") != CLASS_ORDER:
        raise EvaluationError("Unknown/conflicting class mapping")
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise EvaluationError("Require a single input and a single output")
    input_shape, output_shape = tuple(model.inputs[0].shape), tuple(model.outputs[0].shape)
    if input_shape != (None, 350, 350, 3):
        raise EvaluationError(f"Incompatible input shape: {input_shape}; require (None,350,350,3)")
    if output_shape != (None, 3):
        raise EvaluationError(f"Require rank-2 output with exactly 3 probabilities, got {output_shape}")
    for tensor in [*model.inputs, *model.outputs]:
        dtype = getattr(tensor, "dtype", "float32")
        if getattr(dtype, "name", dtype) != "float32":
            raise EvaluationError("Require float32 model input and output")
    layers = [layer for layer in model.layers if type(layer).__name__ != "InputLayer"]
    if len(layers) != 4:
        raise EvaluationError("Require NASNetLarge backbone -> Flatten -> Dense(256) -> Dense(3)")
    backbone, flatten, dense, classifier = layers
    if (type(flatten).__name__ != "Flatten" or type(dense).__name__ != "Dense"
            or type(classifier).__name__ != "Dense" or dense.units != 256 or classifier.units != 3
            or _activation_name(dense) != "relu" or _activation_name(classifier) != "softmax"):
        raise EvaluationError("Model does not have the original Flatten/Dense256/ReLU/Dense3/softmax head")
    if tuple(backbone.output_shape) != (None, 11, 11, 4032):
        raise EvaluationError("Backbone is not the expected 350px NASNetLarge topology")
    for layer in [*layers, *getattr(backbone, "layers", [])]:
        if getattr(layer, "data_format", "channels_last") not in (None, "channels_last"):
            raise EvaluationError("Require channels-last layers")
        policy = getattr(layer, "dtype_policy", None)
        if policy is not None and policy.compute_dtype != "float32":
            raise EvaluationError("Require float32 computation; mixed precision is not the primary contract")
    if model.count_params() != 209813077:
        raise EvaluationError("NASNetLarge/original-head parameter count must be exactly 209813077 (including BN statistics)")
    expected = reference_architecture()
    architecture_hash = validate_architecture_document(loaded_architecture(model), expected["full"])
    trainable = sum(int(np.prod(variable.shape)) for variable in model.trainable_weights)
    nontrainable = sum(int(np.prod(variable.shape)) for variable in model.non_trainable_weights)
    return {"input_shape": list(input_shape), "output_shape": list(output_shape),
            "parameter_count": model.count_params(), "trainable_parameters": trainable,
            "nontrainable_parameters": nontrainable, "class_mapping": CLASS_ORDER,
            "architecture_fingerprint": architecture_hash,
            "architecture_validation": "complete operation/configuration/connectivity comparison; weight-free reference",
            "authentication": "structural checks and supplied provenance; authorship not independently verified"}


def load_paper_model(path: Path, model_format: str, metadata: dict, tf: Any,
                     purpose: str = "paper") -> tuple[Any, dict]:
    """Load a whole model only; no conversion, reconstruction or fallback."""
    with path.open("rb") as stream:
        signature = stream.read(8)
    if signature == b"\x89HDF\r\n\x1a\n":
        detected = "h5"
        import h5py
        with h5py.File(path, "r") as archive:
            if "model_config" not in archive.attrs:
                raise EvaluationError("Weights-only HDF5 is unsupported; exact architecture reconstruction is required")
    elif zipfile.is_zipfile(path):
        detected = "keras"
        with zipfile.ZipFile(path) as archive:
            if not {"config.json", "metadata.json", "model.weights.h5"}.issubset(archive.namelist()):
                raise EvaluationError("Not a supported whole-model .keras archive")
    else:
        raise EvaluationError("Unsupported model container/weights-only checkpoint; supply whole-model .h5 or .keras")
    if model_format not in ("auto", detected):
        raise EvaluationError(f"Model format mismatch: requested {model_format}, found {detected}")
    try:
        model = tf.keras.models.load_model(path, compile=False, safe_mode=True)
    except Exception as error:
        raise EvaluationError(f"Whole-model loading failed (no conversion/fallback): {error}") from error
    result = validate_model_contract(model, metadata, purpose)
    result["format"] = detected
    return model, result


def create_legacy_normalizer(reference: Path) -> tuple[Any, Any]:
    """Fit raw reference directly, exactly as legacy V:38-44; no brightness fit transform."""
    try:
        import staintools
    except ImportError as error:
        raise EvaluationError("Legacy preprocessing requires working staintools and its native dependencies; "
                              "install/configure a verified environment separately. No NumPy fallback.") from error
    target = staintools.read_image(str(reference))
    standardizer = staintools.BrightnessStandardizer()
    normalizer = staintools.StainNormalizer(method="macenko")
    normalizer.fit(target)
    return standardizer, normalizer


def preprocess_legacy(row: dict, path: Path, normalizers: tuple[Any, Any]) -> np.ndarray:
    """RGB load -> PIL Lanczos resize -> brightness -> Macenko -> float32 /255."""
    from PIL import Image
    try:
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["file_sha256"]:
            raise EvaluationError("Image checksum changed after manifest construction")
        with Image.open(io.BytesIO(payload)) as image:
            image = image.convert("RGB")  # No EXIF transpose or colour-profile conversion.
            resized = image.resize((350, 350), Image.Resampling.LANCZOS)
            pixels = np.array(resized)
        standardizer, normalizer = normalizers
        pixels = normalizer.transform(standardizer.transform(pixels))
        result = np.asarray(pixels, dtype=np.float32) / np.float32(255.0)
        if result.shape != (350, 350, 3) or not np.isfinite(result).all() or result.min() < 0 or result.max() > 1:
            raise EvaluationError("Invalid preprocessed image shape/range/finiteness")
        return result
    except Exception as error:
        raise EvaluationError(f"Preprocessing failed for {row['sample_id']}: {error}") from error


def validate_probabilities(probabilities: Any, sample_ids: list[str]) -> tuple[np.ndarray, float]:
    raw = np.asarray(probabilities)
    if raw.shape != (len(sample_ids), 3):
        raise EvaluationError(f"Expected ({len(sample_ids)},3) probabilities for {sample_ids}, got {raw.shape}")
    values = raw.astype(np.float64)  # Lossless float32 promotion; never round or renormalize.
    errors = np.abs(values.sum(axis=1) - 1)
    valid = np.isfinite(values).all(axis=1) & (values >= -1e-6).all(axis=1) & (values <= 1 + 1e-6).all(axis=1) & (errors <= 1e-5)
    if not valid.all():
        index = int(np.flatnonzero(~valid)[0])
        raise EvaluationError(f"Invalid softmax probabilities for {sample_ids[index]}: {values[index].tolist()}")
    return values, float(errors.max(initial=0))


def binary_decision(p_tumour: Any) -> np.ndarray:
    """Strict legacy WSI tumour rule; exactly 0.5 is benign, never argmax."""
    return (np.asarray(p_tumour) > 0.5).astype(np.int64)


def calculate_binary_metrics(labels: Any, probabilities: Any) -> dict:
    """Unweighted binary metrics with tumour positive and unrounded tumour-score AUC."""
    from sklearn.metrics import confusion_matrix, roc_auc_score
    labels, scores = np.asarray(labels), np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels) or not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise EvaluationError("Require aligned finite scores and nonempty binary labels")
    predicted = binary_decision(scores)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None
    recall, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    result = dict(TN=tn, FP=fp, FN=fn, TP=tp, confusion_matrix=matrix.tolist(),
                  label_order=["benign", "tumour"], accuracy=(tp + tn) / len(labels),
                  precision=ratio(tp, tp + fp), recall=recall, sensitivity=recall,
                  tumour_accuracy=recall, specificity=specificity, benign_accuracy=specificity,
                  f1=ratio(2 * tp, 2 * tp + fp + fn),
                  roc_auc=float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
                  exact_threshold_ties=int(np.count_nonzero(scores == 0.5)))
    result["undefined_metrics"] = [key for key, value in result.items() if value is None]
    return result


class Progress:
    def __init__(self, total: int, every: int, seconds: float, logger: logging.Logger,
                 clock: Callable[[], float] = time.perf_counter):
        self.total, self.every, self.seconds, self.logger, self.clock = total, every, seconds, logger, clock
        self.started = self.last_time = clock()
        self.last_count = 0

    def update(self, count: int) -> None:
        now = self.clock()
        if count != self.total and count - self.last_count < self.every and now - self.last_time < self.seconds:
            return
        elapsed, interval = now - self.started, now - self.last_time
        average = count / elapsed if elapsed > 0 else 0.0
        rate = (count - self.last_count) / interval if interval > 0 else 0.0
        eta = f"{(self.total - count) / average:.1f}s" if average else "unavailable"
        self.logger.info("%d/%d (%.1f%%) | loop_elapsed=%.2fs | avg=%.2f img/s | interval=%.2f img/s | ETA=%s",
                         count, self.total, 100 * count / self.total, elapsed, average, rate, eta)
        self.last_count, self.last_time = count, now


def infer_c1(model: Any, manifest: Manifest, preprocess: Callable, batch_size: int,
             timings: Timings, logger: logging.Logger, log_every: int = 1000,
             log_seconds: float = 30) -> tuple[list[dict], dict]:
    """Present each manifest image once; host materialization is inside inference timing."""
    if batch_size < 1:
        raise EvaluationError("Batch size must be positive")
    predictions = []
    calls, max_error = 0, 0.0
    output_dtypes: set[str] = set()
    progress = Progress(len(manifest.rows), log_every, log_seconds, logger, timings.clock)
    for offset in range(0, len(manifest.rows), batch_size):
        rows = manifest.rows[offset:offset + batch_size]
        with timings.measure("preprocessing"):
            batch = np.stack([preprocess(row, manifest.paths[row["sample_id"]]) for row in rows])
        try:
            with timings.measure("inference"):
                output = model(batch, training=False)
                raw = output.numpy() if hasattr(output, "numpy") else np.asarray(output)
        except Exception as error:
            raise EvaluationError(f"C1 inference failed after {len(predictions)} images; "
                                  f"batch samples={[row['sample_id'] for row in rows]}: {error}") from error
        output_dtypes.add(str(np.asarray(raw).dtype))
        calls += 1
        values, error = validate_probabilities(raw, [row["sample_id"] for row in rows])
        max_error = max(max_error, error)
        for row, (gland, nongland, tumour) in zip(rows, values):
            decision = int(binary_decision(tumour))
            result = {key: row[key] for key in PREDICTION_FIELDS if key in row}
            result.update(p_gland=float(gland), p_nongland=float(nongland), p_benign=float(gland + nongland),
                          p_tumour=float(tumour), predicted_binary_code=decision,
                          predicted_binary_label="tumour" if decision else "benign",
                          correct=decision == row["ground_truth_code"])
            predictions.append(result)
        progress.update(len(predictions))
    if len(predictions) != len(manifest.rows):
        raise EvaluationError("Prediction row count does not equal manifest count")
    return predictions, {"cohort_forward_calls": calls, "cohort_image_presentations": len(predictions),
                         "maximum_probability_sum_error": max_error,
                         "model_output_dtypes": sorted(output_dtypes)}


def verify_prediction_export(path: Path, predictions: list[dict], metrics: dict) -> None:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(predictions):
        raise EvaluationError("Prediction export row count mismatch")
    expected = list(csv.DictReader(io.StringIO(csv_bytes(predictions, PREDICTION_FIELDS).decode("utf-8"))))
    for stored, original, serialized in zip(rows, predictions, expected):
        if set(stored) != set(PREDICTION_FIELDS):
            raise EvaluationError("Prediction export columns mismatch")
        for field_name in PREDICTION_FIELDS:
            if stored[field_name] != serialized[field_name]:
                raise EvaluationError(f"Prediction export {field_name} mismatch for {original['sample_id']}")
        if stored["sample_id"] != original["sample_id"]:
            raise EvaluationError("Prediction export order mismatch")
        for field_name in ["p_gland", "p_nongland", "p_benign", "p_tumour"]:
            if float(stored[field_name]) != original[field_name]:
                raise EvaluationError(f"Prediction precision lost for {original['sample_id']}")
        if int(stored["predicted_binary_code"]) != int(binary_decision(float(stored["p_tumour"]))):
            raise EvaluationError("Prediction decision changed on export")
    recomputed = calculate_binary_metrics([int(row["ground_truth_code"]) for row in rows],
                                          [float(row["p_tumour"]) for row in rows])
    if recomputed != metrics:
        raise EvaluationError("Metrics changed after prediction round-trip")


def configure_logging(output: Path, level: str) -> logging.Logger:
    logger = logging.getLogger(f"paper_c1.{uuid.uuid4().hex}")
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)sZ [%(levelname)s] %(message)s")
    formatter.converter = time.gmtime
    for handler in [logging.StreamHandler(), logging.FileHandler(output / "evaluation.log", encoding="utf-8")]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def capture_git() -> dict:
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        try:
            return subprocess.check_output(["git", "--no-optional-locks", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    tracked = git("status", "--porcelain", "--untracked-files=no")
    untracked = git("ls-files", "--others", "--exclude-standard")
    return {"branch": git("branch", "--show-current"), "commit": git("rev-parse", "HEAD"),
            "tracked_dirty": bool(tracked) if tracked is not None else None,
            "untracked_paths": untracked.splitlines() if untracked is not None else None}


def physical_ram_bytes() -> int | None:
    """Best-effort physical RAM inventory using only platform standard-library APIs."""
    try:
        if sys.platform == "win32":
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            *[(name, ctypes.c_ulonglong) for name in ["total_physical", "available_physical",
                              "total_page", "available_page", "total_virtual", "available_virtual", "extended"]]]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_physical)
        elif hasattr(os, "sysconf"):
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (OSError, ValueError, AttributeError):
        pass
    return None


def capture_environment() -> dict:
    versions = {"python": platform.python_version()}
    for name in ["tensorflow", "keras", "numpy", "scikit-learn", "Pillow", "staintools", "h5py", "spams",
                 "spams-bin", "opencv-python", "opencv-python-headless"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    ram = physical_ram_bytes()
    return {"software": versions,
            "platform": {"os": platform.platform(), "kernel": platform.release(),
                         "cpu": platform.processor() or platform.machine(), "ram_bytes": ram,
                         "ram_unavailable_reason": None if ram is not None else "Platform RAM probe unavailable",
                         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}}


def select_device(tf: Any, requested: str) -> tuple[str, dict]:
    visible = tf.config.get_visible_devices("GPU")
    if requested == "gpu" and not visible:
        raise EvaluationError("GPU explicitly requested but no TensorFlow-visible GPU is available")
    if requested == "cpu":
        tf.config.set_visible_devices([], "GPU")
    tf.config.set_soft_device_placement(False)
    selected = "/GPU:0" if requested == "gpu" else "/CPU:0"
    gpu_info = [{"name": item.name, "details": tf.config.experimental.get_device_details(item)} for item in visible]
    device_name = gpu_info[0]["details"].get("device_name", selected) if requested == "gpu" else "CPU"
    return selected, {"selected_device": selected, "device_name": device_name, "tensorflow_visible_gpus": gpu_info,
                      "effective_visible_gpus": [item.name for item in tf.config.get_visible_devices("GPU")],
                      "tensorflow_build": tf.sysconfig.get_build_info()}


def numerical_execution_settings(tf: Any) -> dict:
    """Best-effort observations, not hardware identity or parity claims."""
    def query(path):
        try:
            obj = tf
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj()
        except (AttributeError, RuntimeError):
            return "unavailable"
    return {"tf32_enabled": query("config.experimental.tensor_float_32_execution_enabled"),
            "op_determinism_enabled": query("config.experimental.is_op_determinism_enabled"),
            "optimizer_jit": query("config.optimizer.get_jit"),
            "optimizer_options": query("config.optimizer.get_experimental_options"),
            "intra_op_threads": query("config.threading.get_intra_op_parallelism_threads"),
            "inter_op_threads": query("config.threading.get_inter_op_parallelism_threads"),
            "environment": {key: os.environ.get(key) for key in
                            ["TF_DETERMINISTIC_OPS", "TF_ENABLE_ONEDNN_OPTS", "NVIDIA_TF32_OVERRIDE"]}}


def implementation_identity() -> dict:
    # Full source hash deliberately conservative: even untracked edits are bound.
    # Normalize newlines so a Git CRLF checkout alone does not break CPU/WSL comparison.
    source = Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    preprocessing = "\n".join(inspect.getsource(function).replace("\r\n", "\n") for function in
                              [create_legacy_normalizer, preprocess_legacy])
    return {"version": 2, "evaluator_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "preprocessing_source_sha256": hashlib.sha256(preprocessing.encode()).hexdigest()}


def preprocessing_settings() -> dict:
    # Backend defaults remain unchanged. Bind their installed source bytes, including
    # brightness/Macenko/concentration helpers, instead of silently pinning new values.
    files = {}
    for package in ("staintools", "opencv-python", "spams"):
        try:
            distribution = importlib.metadata.distribution(package)
        except importlib.metadata.PackageNotFoundError:
            files[package] = "unavailable"
            continue
        sources = {str(path).replace("\\", "/"): sha256_file(Path(distribution.locate_file(path)))
                   for path in distribution.files or [] if str(path).endswith(".py")}
        files[package] = fingerprint(sources)
    return {"source_decode": "PIL RGB; no EXIF transpose", "resize_before_normalization": True,
            "brightness": "BrightnessStandardizer.transform; installed defaults",
            "stain_method": "macenko", "stain_options": "installed defaults bound to backend source hashes",
            "reference_fit": "staintools.read_image; no brightness transform",
            "failure_policy": "abort", "backend_source_sha256": files}


def scientific_config(args: argparse.Namespace, model_hash: str, manifest_hash: str,
                      reference_hash: str, versions: dict) -> dict:
    return {"model_sha256": model_hash, "manifest_sha256": manifest_hash, "reference_sha256": reference_hash,
            "cohort": args.cohort, "image_size": args.image_size, "batch_size": args.batch_size,
            "threshold": args.threshold, "comparison_operator": ">", "class_order": CLASS_ORDER,
            "preprocessing": args.preprocessing, "resize": "PIL.Resampling.LANCZOS", "dtype": "float32",
            "scale": "1/255", "strategy": "C1", "software": versions,
            "implementation": implementation_identity(), "preprocessing_settings": preprocessing_settings()}


FINAL_ARTIFACTS = ("manifest.csv", "predictions.csv", "metrics.json", "environment.json", "timing_summary.json")


def completion_document(output: Path, summary: dict) -> dict:
    return {"schema_version": 1, "status": "complete", "run_id": summary["run_id"],
            **{key: summary[key] for key in ("scientific_config_sha256", "model_sha256",
                                            "manifest_sha256", "reference_sha256")},
            "artifacts": {name: sha256_file(output / name) for name in FINAL_ARTIFACTS}}


def validate_completed_run(path: Path, config_hash: str | None = None) -> dict:
    """Only a last-written marker plus intact, matching artifacts constitutes success.

    Accept a run directory, marker, or legacy timing-summary CLI path. The log is
    deliberately not hashed because completion logging occurs after the marker.
    Hashes provide integrity, not authentication against intentional rewriting.
    """
    root = path if path.is_dir() else path.parent
    try:
        marker = json.loads((root / "run_complete.json").read_text(encoding="utf-8"))
        if not isinstance(marker, dict) or marker.get("schema_version") != 1 or marker.get("status") != "complete":
            raise EvaluationError("Invalid completion marker")
        if config_hash is not None and marker.get("scientific_config_sha256") != config_hash:
            raise EvaluationError("Scientific configuration differs from complete baseline")
        if not isinstance(marker.get("run_id"), str) or not marker["run_id"]:
            raise EvaluationError("Invalid completion run ID")
        if not isinstance(marker.get("artifacts"), dict) or set(marker["artifacts"]) != set(FINAL_ARTIFACTS):
            raise EvaluationError("Completion marker missing required artifact hashes")
        for key in ("scientific_config_sha256", "model_sha256", "manifest_sha256", "reference_sha256"):
            if not isinstance(marker.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", marker[key]) is None:
                raise EvaluationError(f"Invalid completion checksum: {key}")
        for name in FINAL_ARTIFACTS:
            digest = marker["artifacts"][name]
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise EvaluationError(f"Invalid artifact checksum: {name}")
            sha256_file(root / name, digest)
        for name in ("metrics.json", "environment.json", "timing_summary.json"):
            document = json.loads((root / name).read_text(encoding="utf-8"))
            state = document["run"] if name == "environment.json" else document
            if state["run_id"] != marker["run_id"] or state["status"] != "complete":
                raise EvaluationError(f"Completion run ID/status mismatch: {name}")
            if document["scientific_config_sha256"] != marker["scientific_config_sha256"]:
                raise EvaluationError(f"Completion scientific fingerprint mismatch: {name}")
        summary = json.loads((root / "timing_summary.json").read_text(encoding="utf-8"))
        for key in ("model_sha256", "manifest_sha256", "reference_sha256"):
            if summary[key] != marker[key]:
                raise EvaluationError(f"Completion identity mismatch: {key}")
        if marker["artifacts"]["manifest.csv"] != marker["manifest_sha256"]:
            raise EvaluationError("Completion manifest identity mismatch")
        return marker
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise EvaluationError(f"Incomplete/invalid completed run at {root}: {error}") from error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--purpose", required=True, choices=["sanity", "paper"])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-format", default="auto", choices=["auto", "h5", "keras"])
    parser.add_argument("--checkpoint-metadata", type=Path)
    parser.add_argument("--class-order")
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--benign-dir", required=True, type=Path)
    parser.add_argument("--tumour-dir", required=True, type=Path)
    parser.add_argument("--stain-reference", required=True, type=Path)
    parser.add_argument("--preprocessing", default="legacy-staintools", choices=["legacy-staintools"])
    parser.add_argument("--image-size", type=int, default=350, choices=[350])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5, choices=[0.5])
    parser.add_argument("--device", required=True, choices=["cpu", "gpu"])
    parser.add_argument("--warmup-batches", type=int, default=0, choices=[0])
    parser.add_argument("--log-level", default="INFO", choices=["INFO", "DEBUG"])
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--log-seconds", type=float, default=30)
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--compare-run", type=Path,
                        help="Prior run directory, run_complete.json or timing_summary.json; require valid final marker and artifact hashes")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    args.class_order = args.class_order.split(",") if args.class_order else None
    if args.batch_size < 1 or args.log_every < 1 or not np.isfinite(args.log_seconds) or args.log_seconds <= 0:
        parser.error("Batch size/log intervals must be positive and finite")
    if args.class_order is not None and args.class_order != CLASS_ORDER:
        parser.error("Require --class-order gland,nongland,tumour")
    if args.purpose == "paper" and (args.checkpoint_metadata is None or args.cohort not in {"VAL1", "VAL2"}):
        parser.error("Paper mode requires checkpoint metadata and cohort VAL1 or VAL2")
    return args


def _execute(args: argparse.Namespace, env: dict, timings: Timings, logger: logging.Logger,
             started: float) -> tuple[dict, dict]:
    with timings.measure("startup"):
        try:
            import tensorflow as tf
        except ImportError as error:
            raise EvaluationError("TensorFlow is required for model evaluation; no installation is performed") from error
        tf.keras.mixed_precision.set_global_policy("float32")
        device, device_info = select_device(tf, args.device)
        env["platform"].update(device_info)
        env["numerical_execution"] = numerical_execution_settings(tf)
        env["software"]["tensorflow"] = getattr(tf, "__version__", env["software"].get("tensorflow"))
    logger.info("Device configuration: %s", device_info)
    with timings.measure("model_load"), tf.device(device):
        model_hash = sha256_file(args.model, args.expected_model_sha256)
        metadata_hash = sha256_file(args.checkpoint_metadata) if args.checkpoint_metadata else None
        metadata = load_checkpoint_metadata(args.checkpoint_metadata, model_hash, args.purpose, args.class_order)
        if args.checkpoint_metadata:
            sha256_file(args.checkpoint_metadata, metadata_hash)
        model, model_info = load_paper_model(args.model, args.model_format, metadata, tf, args.purpose)
        env["model"] = dict(path=str(args.model.resolve()), sha256=model_hash, provenance=metadata, **model_info)
    logger.info("Model loaded | elapsed=%.3fs | sha256=%s", timings.seconds["model_load"], model_hash)
    with timings.measure("manifest"):
        manifest = build_manifest(args.benign_dir, args.tumour_dir, args.cohort)
        manifest_hash = manifest.sha256
        if args.expected_manifest_sha256 and manifest_hash != args.expected_manifest_sha256.lower():
            raise EvaluationError("Expected dataset manifest SHA-256 mismatch")
    counts = manifest.counts
    logger.info("Manifest ready | total=%d | benign=%d | tumour=%d | sha256=%s", counts["total"], counts["benign"], counts["tumour"], manifest_hash)
    if manifest.duplicate_content:
        logger.warning("Identical-content entries retained: %s", manifest.duplicate_content)
    logger.debug("Ignored non-image files: %s", manifest.ignored_files)
    env["dataset"] = dict(roots={key: str(value) for key, value in manifest.roots.items()},
                          manifest_sha256=manifest_hash, counts=counts, duplicate_content=manifest.duplicate_content,
                          ignored_files=manifest.ignored_files, cohort_completeness_verified=False)
    with timings.measure("reference_setup"):
        reference_hash = sha256_file(args.stain_reference)
        normalizers = create_legacy_normalizer(args.stain_reference)
    env["preprocessing"] = dict(image_size=350, resize="PIL.Resampling.LANCZOS", dtype="float32",
                                scale="1/255", backend="legacy-staintools", failure_policy="abort",
                                reference_path=str(args.stain_reference.resolve()), reference_sha256=reference_hash,
                                sequence=["RGB decode", "resize", "brightness", "Macenko", "float32 /255"],
                                historical_numerical_parity_verified=False)
    config = scientific_config(args, model_hash, manifest_hash, reference_hash, env["software"])
    config_hash = fingerprint(config)
    if args.compare_run:
        validate_completed_run(args.compare_run, config_hash)
    env["scientific_configuration"] = config
    env["scientific_config_sha256"] = config_hash
    with timings.measure("output"):
        atomic_write(args.output / "manifest.csv", csv_bytes(manifest.rows, MANIFEST_FIELDS))
        write_json(args.output / "environment.json", env)
    logger.info("Starting C1 | batch_size=%d | device=%s", args.batch_size, device)
    with tf.device(device):
        predictions, accounting = infer_c1(model, manifest,
                                           lambda row, path: preprocess_legacy(row, path, normalizers),
                                           args.batch_size, timings, logger, args.log_every, args.log_seconds)
    logger.info("C1 completed | processed=%d", len(predictions))
    with timings.measure("metrics"):
        metrics = calculate_binary_metrics([row["ground_truth_code"] for row in predictions],
                                           [row["p_tumour"] for row in predictions])
    logger.info("Metrics calculated: %s", metrics)
    with timings.measure("output"):
        atomic_write(args.output / "predictions.csv", csv_bytes(predictions, PREDICTION_FIELDS))
    with timings.measure("integrity"):
        verify_prediction_export(args.output / "predictions.csv", predictions, metrics)
        validate_manifest(manifest)
        sha256_file(args.output / "manifest.csv", manifest_hash)
        sha256_file(args.model, model_hash)
        sha256_file(args.stain_reference, reference_hash)
        if args.checkpoint_metadata:
            sha256_file(args.checkpoint_metadata, metadata_hash)
    env["evaluation"] = dict(threshold=0.5, comparison_operator=">", positive_class="tumour",
                              exact_threshold_ties=metrics["exact_threshold_ties"], batch_size=args.batch_size,
                              inference_strategy="C1", **accounting)
    total = timings.clock() - started
    report = timings.report(len(predictions), total)
    summary = dict(schema_version=1, run_id=env["run"]["run_id"], status="complete", purpose=args.purpose,
                   cohort=args.cohort, device=device_info.get("device_name", device), tensorflow_device=device,
                   tensorflow_visible_gpus=device_info["tensorflow_visible_gpus"],
                   samples=len(predictions), benign_samples=counts["benign"], tumour_samples=counts["tumour"],
                    batch_size=args.batch_size, model_sha256=model_hash, manifest_sha256=manifest_hash,
                    reference_sha256=reference_hash,
                   scientific_config_sha256=config_hash, timing=report, timing_scope=TIMING_SCOPE)
    results = dict(schema_version=1, status="complete", run_id=env["run"]["run_id"],
                   scientific=metrics, scientific_config_sha256=config_hash,
                   performance=dict(counts=counts, timing=report, timing_scope=TIMING_SCOPE,
                                    throughput={key: value for key, value in report.items() if key.endswith("images_per_second")}))
    return results, summary


def run(args: argparse.Namespace) -> int:
    timings = Timings()
    started = timings.clock()
    started_at_utc = utc_now()
    env: dict = {}
    with timings.measure("startup"):
        git = capture_git()  # Before creating our output files.
        if args.output.exists():
            raise EvaluationError(f"Output directory already exists; will not overwrite: {args.output}")
        target = args.output.resolve()
        for root in [args.benign_dir.resolve(), args.tumour_dir.resolve()]:
            if target == root or root in target.parents or target in root.parents:
                raise EvaluationError("Output and input dataset directories must not overlap")
        args.output.mkdir(parents=True)
        logger = configure_logging(args.output, args.log_level)
    try:
        with timings.measure("startup"):
            env = capture_environment()
            env.update(git=git, run=dict(run_id=uuid.uuid4().hex, purpose=args.purpose, cohort=args.cohort,
                                        started_at_utc=started_at_utc, status="running", command=sys.argv,
                                        configuration={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}))
        logger.info("Starting %s native C1 evaluation | configuration=%s | git=%s | environment=%s",
                    args.cohort, env["run"]["configuration"], git, env)
        results, summary = _execute(args, env, timings, logger, started)
        env["run"].update(status="complete", completed_at_utc=utc_now())
        env["timing"] = summary["timing"]
        # Individual status fields are provisional until the LAST marker validates.
        write_json(args.output / "environment.json", env)
        write_json(args.output / "timing_summary.json", summary)
        write_json(args.output / "metrics.json", results)
        write_json(args.output / "run_complete.json", completion_document(args.output, summary))
        validate_completed_run(args.output, summary["scientific_config_sha256"])
        logger.info("Timing results: %s", summary)
        logger.info("Evaluation complete | total=%.3fs | output=%s", summary["timing"]["total_seconds"], args.output)
        return 0
    except (Exception, KeyboardInterrupt):
        logger.exception("Evaluation failed; no complete metrics artifact")
        env.setdefault("run", {}).update(status="failed", completed_at_utc=utc_now())
        env["partial_timing"] = {**{f"{key}_seconds": value for key, value in timings.seconds.items()},
                                 "total_seconds": timings.clock() - started, "status": "partial"}
        logger.info("Partial timing (not a completed benchmark): %s", env["partial_timing"])
        # These paths belong to this newly-created run only; preserve log and useful partial CSVs.
        for name in ["run_complete.json", "metrics.json", "timing_summary.json"]:
            path = args.output / name
            if path.exists():
                path.unlink()
        try:
            write_json(args.output / "environment.json", env)
        except Exception:
            logger.exception("Could not persist failed environment metadata")
        return 1
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except (EvaluationError, OSError) as error:
        logging.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
