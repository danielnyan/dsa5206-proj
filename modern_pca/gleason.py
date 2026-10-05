"""Shared tumour-only contracts; heavyweight libraries are imported on demand."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

CLASSES = ['GP3', 'GP4', 'GP5']
SCHEDULE = [('normal_conv_1_17', 7, 1e-5)] + [
    (f'normal_conv_1_{n}', 1, 1e-6) for n in (14, 11, 9, 7, 5, 3, 1)]
ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / '4_WSI_pipeline/WSI_pipeline_v6/images/standard_he_stain_small.jpg'
REFERENCE_SHA = '1d31d575c21dcbcf172957b0783c37fcb07872d3129b3a7b3e86ede25ea0a1c1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def provenance():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    try:
        code = {'commit': git('rev-parse', 'HEAD'), 'status': git('status', '--porcelain')}
    except (subprocess.CalledProcessError, FileNotFoundError):
        code = {'commit': 'packaged-source', 'status': 'see RELEASE.json'}
    return dict(utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
                python=sys.version, platform=platform.platform(), code=code,
                source_sha256={p.name: sha(p) for p in Path(__file__).parent.glob('*gleason*.py')})


def write_csv(path, rows, fields=None):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(manifests, dataset, split):
    root = Path(manifests)
    complete = json.loads((root.parent / 'prepared.json').read_text())
    filename = f'{dataset}_{split}.csv'
    if sha(root / filename) != complete['manifests'][filename]:
        raise ValueError(f'Manifest checksum mismatch: {filename}')
    with (root / filename).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        label = int(row['label'])
        if row['dataset'] != dataset or row['split'] != split or not 0 <= label < 3:
            raise ValueError('Invalid manifest class/dataset/split')
        if row['class_name'] != CLASSES[label]:
            raise ValueError('Class order mismatch')
        row['label'] = label
    return rows


def cache_path(cache, row):
    root = Path(cache).resolve()
    path = (root / row['cache_file']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Cache path escapes root')
    return path


def verify_cache(cache, rows):
    for row in rows:
        if sha(cache_path(cache, row)) != row['cache_sha256']:
            raise ValueError(f"Cache checksum mismatch: {row['sample_id']}")


def dataset(rows, cache, batch, training=False, seed=42):
    import tensorflow as tf
    paths = [str(cache_path(cache, row)) for row in rows]
    ds = tf.data.Dataset.from_tensor_slices((paths, [r['label'] for r in rows]))
    if training:
        ds = ds.shuffle(len(rows), seed=seed, reshuffle_each_iteration=True)
    def decode(path, label):
        image = tf.io.decode_png(tf.io.read_file(path), channels=3)
        image = tf.ensure_shape(image, (350, 350, 3))
        image = tf.cast(image, tf.float32) / 255.
        if training:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
        return image, tf.one_hot(label, 3)
    return ds.map(decode, num_parallel_calls=2).batch(batch).prefetch(1)


def gpu_setup(seed):
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError('This operation requires a GPU allocation')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.utils.set_random_seed(seed)
    return tf


def load_model(model_dir, allow_smoke=False):
    import tensorflow as tf
    root = Path(model_dir)
    meta = json.loads((root / 'run.json').read_text())
    if meta.get('complete') is not True or meta['class_order'] != CLASSES:
        raise ValueError('Incomplete model or wrong output classes')
    if meta['smoke'] and not allow_smoke:
        raise ValueError('Smoke checkpoint cannot be used for final evaluation')
    if sha(root / 'final.keras') != meta['model_sha256']:
        raise ValueError('Model checksum mismatch')
    model = tf.keras.models.load_model(root / 'final.keras', compile=False)
    if model.input_shape != (None, 350, 350, 3) or model.output_shape != (None, 3):
        raise ValueError('Wrong model shape')
    return model, meta


def legacy_score(percentages):
    """Execute the released pure scoring function, without its import-time models.

    AST extraction avoids duplicating the scoring policy or importing the legacy
    script's top-level training/inference operations.
    """
    import ast
    values = tuple(float(x) for x in percentages)
    import math
    if len(values) != 3 or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('Expected three finite nonnegative percentages')
    if not math.isclose(sum(values), 100., abs_tol=0.2):
        raise ValueError('Percentages must sum to approximately 100')
    if not hasattr(legacy_score, '_function'):
        path = ROOT / '5_Gleason_score_minimal_tumor_size/GS_number_of_patches.py'
        function = next(n for n in ast.parse(path.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'gscoring')
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        legacy_score._function = namespace['gscoring']
    primary, secondary, isup = legacy_score._function(values)
    return dict(primary=primary, secondary=secondary, gleason_sum=primary+secondary, isup=isup)
