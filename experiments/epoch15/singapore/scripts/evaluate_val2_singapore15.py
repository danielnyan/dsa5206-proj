"""Independent VAL2 evaluation of the Vanda-trained binary NASNetLarge.

Requires execution from repository root as: python -m modern_pca.vanda_val2_batch32.
No training, model selection, augmentation, calibration or threshold optimization.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import sys

import numpy as np

from modern_pca import reimplementation as r
from modern_pca.evaluate_paper import calculate_binary_metrics, binary_decision


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    os.replace(tmp, path)


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--training-summary', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--extracted', type=Path, required=True)
    p.add_argument('--stain-reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, choices=[32], default=32)
    p.add_argument('--log-every', type=int, default=1024)
    return p.parse_args()


def main():
    args = get_args()
    start = time.perf_counter()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError('Output directory is not empty: ' + str(args.output))
    r.ensure_output_separation(args.extracted, [args.output])
    for p in (args.model, args.training_summary,
              args.manifest, args.stain_reference):
        if not p.is_file():
            raise FileNotFoundError(str(p))


    summary = json.loads(args.training_summary.read_text())
    model_digest = sha256(args.model)

    if summary.get("checkpoint_sha256") != model_digest:
        raise RuntimeError("Singapore epoch-15 model SHA256 mismatch")
    if summary.get("samples") != 11020:
        raise RuntimeError("Unexpected Singapore training count")
    if summary.get("updates") != 111:
        raise RuntimeError("Unexpected training update count")
    if summary.get("source_optimizer_iterations") != 1167:
        raise RuntimeError("Unexpected starting Adam iterations")
    if summary.get("final_optimizer_iterations") != 1278:
        raise RuntimeError("Unexpected final Adam iterations")
    if summary.get("learning_rate") != 1e-6:
        raise RuntimeError("Unexpected learning rate")
    if summary.get("trainable_boundary") != "normal_conv_1_1":
        raise RuntimeError("Unexpected fine-tuning boundary")

    print("Singapore epoch-15 model verified:", model_digest, flush=True)

    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) != 1:
        raise RuntimeError(f'Expected exactly 1 visible A40 GPU, got {len(gpus)} GPUs')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.experimental.set_synchronous_execution(True)
    tf.keras.mixed_precision.set_global_policy('float32')
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    print('GPU:', gpus[0], flush=True)

    with tf.device('/GPU:0'):
        model = tf.keras.models.load_model(args.model, compile=False)
        r.validate_model(model)
        model.trainable = False
    weights_before = r.weight_hashes(model.weights)

    rows = r.read_manifest(args.manifest, 'VAL2')
    if len(rows) != 34103:
        raise RuntimeError(f'Expected 34103 VAL2 rows; got {len(rows)}')
    ids = [x['sample_id'] for x in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError('Duplicate sample IDs')
    paths = [r.image_path(row, args.extracted, final_evaluation=True) for row in rows]
    print('Verifying VAL2 source image sizes and SHA256 hashes...', flush=True)
    verify_start = time.perf_counter()
    for i, (row, path) in enumerate(zip(rows, paths), 1):
        if not path.is_file():
            raise FileNotFoundError(str(path))
        if path.stat().st_size != int(row['file_size_bytes']):
            raise RuntimeError('Image size mismatch: ' + str(path))
        if sha256(path) != row['file_sha256']:
            raise RuntimeError('Image digest mismatch: ' + str(path))
        if i % 5000 == 0:
            print(f'Image verification {i}/{len(rows)}', flush=True)
    print('Image SHA256 verification complete, sec=', round(time.perf_counter()-verify_start, 2), flush=True)

    reference = r.normalizers(args.stain_reference)
    args.output.mkdir(parents=True, exist_ok=True)

    # Scientific input pipeline identical to baseline evaluation: one preprocessing per row.
    # First 32 VAL2 images are used for *numerical batch invariance*, not model selection.
    first_input = np.stack([r.preprocess_legacy(row, path, reference)
                            for row, path in zip(rows[:32], paths[:32])])
    with tf.device('/GPU:0'):
        single = np.concatenate([model(first_input[i:i+1], training=False).numpy()
                                 for i in range(32)], axis=0)
        batched = model(first_input, training=False).numpy()
    max_abs = float(np.max(np.abs(single.astype(np.float64)-batched.astype(np.float64))))
    single_decisions = np.asarray([int(binary_decision(float(v[1]))) for v in single])
    batch_decisions = np.asarray([int(binary_decision(float(v[1]))) for v in batched])
    if not np.array_equal(single_decisions, batch_decisions):
        raise RuntimeError('Batch32 vs batch1 changed classification for first 32 images')
    if max_abs > 1e-4:
        raise RuntimeError(f'Batch32 vs batch1 exceeds probability tolerance: {max_abs}')
    print(f'BATCH1/BATCH32 EQUIVALENCE PASS max_abs={max_abs:.8g}, decisions=32/32', flush=True)
    atomic_json(args.output / 'batch_equivalence.json', {
        'samples':32, 'max_probability_abs_difference':max_abs,
        'all_decisions_identical':True, 'tolerance':1e-4,
        'purpose':'numerical batch invariance check, not model selection'
    })

    fields = ['row_index', 'sample_id', 'ground_truth_code', 'p_benign', 'p_tumour', 'predicted_binary_code']
    t_pre = t_infer = 0.0
    scores = []
    pfile = args.output / 'predictions.csv'
    with pfile.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for offset in range(0, len(rows), args.batch_size):
            batch_rows = rows[offset:offset+args.batch_size]
            batch_paths = paths[offset:offset+args.batch_size]
            t0 = time.perf_counter()
            if offset == 0:
                x = first_input
            else:
                x = np.stack([r.preprocess_legacy(row, path, reference)
                              for row, path in zip(batch_rows, batch_paths)])
            t_pre += time.perf_counter() - t0
            t0 = time.perf_counter()
            with tf.device('/GPU:0'):
                predictions = np.asarray(model(x, training=False).numpy(), dtype=np.float64)
            t_infer += time.perf_counter() - t0
            if predictions.shape != (len(batch_rows), 2) or not np.isfinite(predictions).all():
                raise RuntimeError('Invalid inference output shape or nonfinite probability')
            if np.any(predictions < -1e-6) or np.any(predictions > 1+1e-6) or np.any(np.abs(predictions.sum(axis=1)-1)>1e-5):
                raise RuntimeError('Model probabilities invalid')
            for row, p in zip(batch_rows, predictions):
                score = float(p[1])
                scores.append(score)
                writer.writerow({
                    'row_index':row['row_index'], 'sample_id':row['sample_id'],
                    'ground_truth_code':row['ground_truth_code'],
                    'p_benign':float(p[0]), 'p_tumour':score,
                    'predicted_binary_code':int(binary_decision(score)),
                })
            done = offset + len(batch_rows)
            if done % args.log_every < args.batch_size or done == len(rows):
                print(f'VAL2 INFERENCE {done}/{len(rows)} elapsed_min={(time.perf_counter()-start)/60:.2f}', flush=True)

    if len(scores) != len(rows):
        raise RuntimeError('Coverage error')
    metrics = calculate_binary_metrics([row['ground_truth_code'] for row in rows], scores)
    if weights_before != r.weight_hashes(model.weights):
        raise RuntimeError('Model weights mutated during evaluation')
    if model_digest != sha256(args.model):
        raise RuntimeError('Model file mutated during evaluation')
    result = {
        'purpose':'vanda_epoch15_Singapore_all11020_VAL2_binary_evaluation',
        'model':str(args.model.resolve()), 'model_sha256':model_digest,
        'training_summary_sha256':sha256(args.training_summary),
        'training_data':'Singapore 11020 eligible gland patches',
        'manifest_sha256':sha256(args.manifest),
        'stain_reference_sha256':sha256(args.stain_reference),
        'samples':len(rows), 'batch_size':32, 'threshold':'p_tumour > 0.5',
        'class_order':['benign','tumour'], 'metrics':metrics,
        'training_performed':False,
        'timing_seconds':{
            'preprocessing':t_pre, 'inference':t_infer,
            'end_to_end_including_hash_verification':time.perf_counter()-start
        }
    }
    atomic_json(args.output / 'metrics.json', result)
    atomic_json(args.output / 'confusion_matrix.json', {
        'matrix':metrics['confusion_matrix'], 'label_order':['benign','tumour'],
        'orientation':'rows=actual, columns=predicted'
    })
    atomic_json(args.output / 'run_complete.json', {
        'status':'PASS', 'samples':len(rows),
        'model_sha256':model_digest,
        'artifacts_sha256':{name:sha256(args.output / name) for name in
                           ('metrics.json','predictions.csv','confusion_matrix.json','batch_equivalence.json')}
    })
    print('=== VAL2 EVALUATION PASS ===', flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    print('Results:', args.output, flush=True)


if __name__ == '__main__':
    main()
