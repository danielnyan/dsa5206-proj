"""Original VAL1-domain retention evaluation using the epoch-14 verified image cache.

The cache is exactly the data input used during baseline validation; no new image
preprocessing, no augmentation, no optimizer, and no checkpoint mutation.
"""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from . import preprocessing_cache as c
from . import reimplementation as r
from .vanda_cache_consumer import open_verified_cache
from .vanda_sicap_dev_eval import metrics as binary_metrics

BASE = Path('/scratch/e1536052/DSA5206')
MODELS = {
    'epoch14': (BASE / 'vanda_runs/nasnet_production_14epoch/epoch14.keras',
                '433dd48aae800c659d1cba092af2859499bb804d2228ff978a6e9c9602b2adde'),
    'head_only': (BASE / 'vanda_runs/sicap_head_only_epoch15/epoch15_head_only.keras',
                  '4d5b2127484f1e2c37ff33c3bd7ca451f77984d47b4f43b09f9710c6fd64dfd7'),
}
EXPECTED = 29164


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-key', required=True, choices=sorted(MODELS))
    p.add_argument('--cache', required=True, type=Path)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()
    if args.batch_size < 1 or args.batch_size > 32:
        raise ValueError('Batch size must be 1..32')
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite existing output directory')
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    model_path, expected_sha = MODELS[args.model_key]
    if sha256(model_path) != expected_sha:
        raise RuntimeError('Model file SHA mismatch')
    start = time.perf_counter()

    # Verify the source validation manifest and its labels before accessing cache.
    with args.manifest.open(newline='') as f:
        manifest_rows = list(csv.DictReader(f))
    if len(manifest_rows) != EXPECTED:
        raise RuntimeError(f'Expected {EXPECTED} manifest records, got {len(manifest_rows)}')
    labels_by_id = {}
    for row in manifest_rows:
        sample_id = row['sample_id']
        if sample_id in labels_by_id or row['cohort'] != 'VAL1':
            raise RuntimeError('Duplicate sample ID or unexpected cohort')
        label = int(row['ground_truth_code'])
        if label not in (0, 1):
            raise RuntimeError('Invalid class label')
        labels_by_id[sample_id] = label

    # Reuse verified, stain-normalized 350x350 float32 tensors that the original
    # Vanda trainer used for its internal validation. This avoids accidental
    # pipeline differences and does not re-read/re-normalize raw image files.
    reader = open_verified_cache(args.cache)
    indices = reader.indices('internal_validation')
    if len(indices) != EXPECTED or len(set(indices)) != EXPECTED:
        raise RuntimeError('Cache validation indices mismatch')

    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) != 1:
        raise RuntimeError(f'Exactly one GPU required; got {len(gpus)}')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.keras.mixed_precision.set_global_policy('float32')
    tf.keras.utils.set_random_seed(42)

    model = tf.keras.models.load_model(model_path, compile=False)
    r.validate_model(model)
    model.trainable = False
    weights_before = r.weight_hashes(model.weights)

    dataset = c.dataset(reader, indices, batch_size=args.batch_size,
                        parallel=1, prefetch=1, augment=False, epoch=14)
    args.output.mkdir(parents=True, exist_ok=False)
    scores = []
    labels = []
    seen_indices = []
    elapsed_preparation = time.perf_counter() - start
    print(f'VAL1 RETENTION START model={args.model_key} samples={EXPECTED} '
          f'cache={args.cache} model_sha256={expected_sha}', flush=True)
    fields = ['cache_index', 'ground_truth_code', 'p_benign', 'p_tumour', 'predicted_binary_code']
    with (args.output / 'predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for x, y, original_indices in dataset:
            batch_indices = [int(v) for v in original_indices.numpy().tolist()]
            if batch_indices != indices[len(seen_indices):len(seen_indices) + len(batch_indices)]:
                raise RuntimeError('Validation cache index order mismatch')
            values = np.asarray(model(x, training=False).numpy(), dtype=np.float64)
            truth = np.asarray(y.numpy(), dtype=np.int32)
            if values.shape != (len(truth), 2) or not np.isfinite(values).all():
                raise RuntimeError('Nonfinite or incorrectly shaped model probabilities')
            if not np.allclose(values.sum(axis=1), 1, atol=1e-5, rtol=0) or (values < -1e-6).any() or (values > 1 + 1e-6).any():
                raise RuntimeError('Invalid softmax probabilities')
            for idx, actual, prob in zip(batch_indices, truth, values):
                actual = int(actual)
                if actual not in (0, 1):
                    raise RuntimeError('Invalid cache class label')
                score = float(prob[1])
                labels.append(actual)
                scores.append(score)
                writer.writerow({'cache_index': idx, 'ground_truth_code': actual,
                                 'p_benign': float(prob[0]), 'p_tumour': score,
                                 'predicted_binary_code': int(score > 0.5)})
            seen_indices.extend(batch_indices)
            if len(labels) % 320 < args.batch_size or len(labels) == EXPECTED:
                print(f'VAL1 RETENTION {len(labels)}/{EXPECTED} '
                      f'elapsed_min={(time.perf_counter()-start)/60:.2f}', flush=True)
    if len(labels) != EXPECTED or len(seen_indices) != EXPECTED:
        raise RuntimeError('Incomplete validation coverage')
    if r.weight_hashes(model.weights) != weights_before:
        raise RuntimeError('Evaluation changed model weights')
    if sha256(model_path) != expected_sha:
        raise RuntimeError('Model file changed during evaluation')

    results = binary_metrics(np.asarray(labels, dtype=np.int32),
                             np.asarray(scores, dtype=np.float64))
    metadata = {
        'purpose': 'VAL1_internal_validation_original_domain_retention_diagnostic',
        'limitation': 'VAL1 internal validation was monitored during baseline training; not an independent holdout',
        'model_key': args.model_key, 'model_sha256': expected_sha,
        'manifest_sha256': sha256(args.manifest),
        'cache_manifest_sha256': sha256(args.cache / 'cache.json'),
        'samples': EXPECTED, 'batch_size': args.batch_size,
        'threshold': 'p_tumour > 0.5', 'augmentation': False,
        'data_source': 'verified original VAL1 preprocessed cache',
        'training_performed': False,
        'metrics': results,
        'timing_seconds': {'initialization': elapsed_preparation,
                           'end_to_end': time.perf_counter() - start}
    }
    (args.output / 'metrics.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (args.output / 'run_complete.json').write_text(json.dumps({
        'status': 'PASS', 'artifacts_sha256': {
            name: sha256(args.output / name) for name in ('metrics.json', 'predictions.csv')
        }
    }, indent=2) + '\n')
    print('=== VAL1 RETENTION EVALUATION PASS ===', flush=True)
    print(json.dumps(results, indent=2), flush=True)
    print('Results:', args.output, flush=True)

if __name__ == '__main__':
    main()
