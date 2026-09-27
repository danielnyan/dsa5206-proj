from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .stain import MacenkoNormalizer


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass
class DatasetBundle:
    train: tf.data.Dataset
    validation: tf.data.Dataset
    test: tf.data.Dataset
    class_names: list[str]
    counts: dict[str, int]


def indexed_files(root: str | Path) -> tuple[list[str], np.ndarray, list[str]]:
    root = Path(root)
    classes = sorted(path.name for path in root.iterdir() if path.is_dir())
    if len(classes) < 2:
        raise ValueError(f"Expected at least two class directories under {root}, found {classes}")
    files: list[str] = []
    labels: list[int] = []
    for label, class_name in enumerate(classes):
        selected = sorted(
            str(path) for path in (root / class_name).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        files.extend(selected)
        labels.extend([label] * len(selected))
    if not files:
        raise ValueError(f"No supported images found under {root}")
    return files, np.asarray(labels, dtype=np.int32), classes


def _decode(path, label, image_size: int, training: bool, normalizer):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    image = tf.image.resize(image, (image_size, image_size), antialias=True)
    if normalizer is not None:
        image = tf.numpy_function(
            lambda value: normalizer.transform(value.astype(np.uint8)).astype(np.float32),
            [image],
            tf.float32,
        )
        image.set_shape((image_size, image_size, 3))
    image = tf.cast(image, tf.float32) / 255.0
    if training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
    return image, label


def _dataset(files, labels, image_size, batch_size, training, seed, normalizer):
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    if training:
        ds = ds.shuffle(len(files), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda path, label: _decode(path, label, image_size, training, normalizer),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not training,
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_datasets(
    root: str | Path,
    image_size: int = 350,
    batch_size: int = 8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    max_samples: int | None = None,
    stain_reference: str | Path | None = None,
) -> DatasetBundle:
    files, labels, classes = indexed_files(root)
    if max_samples and max_samples < len(files):
        keep, _ = train_test_split(
            np.arange(len(files)), train_size=max_samples, stratify=labels, random_state=seed
        )
        files = [files[index] for index in keep]
        labels = labels[keep]
    train_files, other_files, train_labels, other_labels = train_test_split(
        files,
        labels,
        test_size=validation_fraction + test_fraction,
        stratify=labels,
        random_state=seed,
    )
    relative_test = test_fraction / (validation_fraction + test_fraction)
    val_files, test_files, val_labels, test_labels = train_test_split(
        other_files,
        other_labels,
        test_size=relative_test,
        stratify=other_labels,
        random_state=seed,
    )
    counts = {"train": len(train_files), "validation": len(val_files), "test": len(test_files)}
    normalizer = MacenkoNormalizer(stain_reference) if stain_reference else None
    return DatasetBundle(
        _dataset(train_files, train_labels, image_size, batch_size, True, seed, normalizer),
        _dataset(val_files, val_labels, image_size, batch_size, False, seed, normalizer),
        _dataset(test_files, test_labels, image_size, batch_size, False, seed, normalizer),
        classes,
        counts,
    )
