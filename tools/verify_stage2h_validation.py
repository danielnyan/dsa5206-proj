"""Read-only epoch-8 restoration and bounded VAL1 validation batch comparison.

No production entry point, training microstep, optimizer update or cache writes.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time
import subprocess
import weakref

import numpy as np

from modern_pca import cached_training as t, preprocessing_cache as c, reimplementation as r
from modern_pca.evaluate_paper import binary_decision, calculate_binary_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-batch', type=int, choices=(2, 4), default=4)
    comparison = parser.parse_args().comparison_batch
    logging.basicConfig(level=logging.INFO)
    blocked = []
    def audit(event, args):
        if event not in ('open', 'os.listdir', 'os.scandir') or not args:
            return
        if not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = str(Path(os.fsdecode(args[0])).absolute()).lower()
        if 'val2' in path or 'val_dataset_2' in path:
            blocked.append(path)
            raise RuntimeError('VAL2 access forbidden in validation verification')
    sys.addaudithook(audit)
    source = Path('runs/stage2h_production')
    resume = source / 'epoch_08_resume'
    output = Path('reports/stage2h_validation_batch_verification.json' if comparison == 4
                  else 'reports/stage2h_validation_batch2_verification.json')
    fresh = Path(f'runs/stage2h_production_resume_epoch08_val{1 if comparison == 4 else 2}')
    assert not fresh.exists(), 'Intended resume output already exists'
    # Inventory only the explicit original production run, never source images.
    files = [p for p in source.rglob('*') if p.is_file()]
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in files}
    json_before = {str(p): r.sha256_file(p) for p in files if p.suffix == '.json'}
    cache = Path('runs/stage2g_full_cache/cache')
    cache_before = {str(cache / n): r.sha256_file(cache / n)
                    for n in ('cache.json', 'context.json', 'verification.json')}
    report = dict(status='STARTED', checkpoint=str(resume), production_training_started=False,
                  training_batches=0, optimizer_updates=0,
                  probability_tolerance=dict(atol=1e-5, rtol=1e-5), metric_absolute_tolerance=1e-6)
    try:
        report['nvidia_smi_before'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv'], text=True)
        import tensorflow as tf
        def forbidden(*args, **kwargs):
            raise AssertionError('Training/update prohibited during validation verification')
        tf.keras.optimizers.Adam.apply_gradients = forbidden
        r.Accumulator.add = forbidden
        r.Accumulator.flush = forbidden
        t.train_batches = forbidden
        device = r.gpu_setup()
        memory = lambda: tf.config.experimental.get_memory_info('GPU:0')
        reader = t.checked_reader(cache)
        logging.info('Cache checked; building model for strict epoch-8 restore')
        model, counts = r.build_model()
        r.configure_stage(model, r.SCHEDULE[1][0])
        optimizer = r.adam(1e-6)
        cache_sha = r.sha256_file(cache / 'cache.json')
        state = t.restore_checkpoint(model, optimizer, resume, cache_sha, 'production')
        saved = t.f.read(resume / 'checkpoint.json')
        assert state['completed_epochs'] == 8 and state['boundary'] == r.SCHEDULE[1][0]
        assert state['rate'] == 1e-6
        assert [entry['epoch'] for entry in state['history']] == list(range(1, 9))
        assert t.f.read(source / 'history.json')['epochs'] == state['history']
        assert int(optimizer.iterations.numpy()) == 1167
        report.update(device=device, strict_checkpoint_restore='PASS', restored_optimizer_iterations=1167,
                      saved_history_epochs=list(range(1, 9)), next_epoch=9, next_boundary=r.SCHEDULE[2][0],
                      cache_compatibility=reader.environment_compatibility)
        snapshots = {'restored_epoch8': memory()}
        old_ref = weakref.ref(optimizer)
        del optimizer
        gc.collect()
        snapshots['after_old_adam_collection'] = memory()
        report['old_adam_collected'] = old_ref() is None
        # Match the next stage's state allocation, without running/tracing training.
        r.configure_stage(model, r.SCHEDULE[2][0])
        optimizer = r.adam(1e-6); optimizer.build(model.trainable_variables)
        accumulator = r.Accumulator(model, optimizer)
        snapshots['epoch9_adam_and_accumulator_allocated'] = memory()
        assert int(optimizer.iterations.numpy()) == 0 and accumulator.count == 0
        logging.info('Epoch-8 restore passed; comparing 128 deterministic internal-validation images')
        # Fixed hash-ranked 64/class subset, selected without looking at predictions.
        indices = reader.indices('internal_validation')
        selected = []
        for label in r.CLASSES:
            candidates = [i for i in indices if reader.items[i]['class_name'] == label]
            selected.extend(sorted(candidates, key=lambda i: r.rank_id(
                reader.items[i]['sample_id'], 'stage2h-validation-batch-equivalence|42|'))[:64])
        selected.sort()  # preserve the full-validation relative order
        previous = t.f.read(Path('reports/stage2h_validation_batch_verification.json'))
        assert [reader.items[i]['sample_id'] for i in selected] == previous['sample_ids']
        results = {}; timings = {}
        for size in (comparison, 1):
            # Warm up each shape before timing the identical complete subset.
            t.validation(model, c.dataset(reader, selected[:4], batch_size=size,
                                          parallel=1, prefetch=2, augment=False, epoch=9))
            tf.config.experimental.reset_memory_stats('GPU:0')
            started = time.perf_counter()
            results[size] = t.validation(model, c.dataset(reader, selected, batch_size=size,
                                                         parallel=1, prefetch=2, augment=False, epoch=9))
            timings[size] = time.perf_counter() - started
            snapshots[f'validation_batch_{size}'] = memory()
            assert results[size]['indices'] == selected
            logging.info('Validation batch %d complete', size)
        assert results[1]['labels'] == results[comparison]['labels']
        one, four = (np.asarray(results[size]['scores']) for size in (1, comparison))
        delta = np.abs(one - four)
        metrics = {size: calculate_binary_metrics(results[size]['labels'], results[size]['scores'])
                   for size in (1, comparison)}
        metric_deltas = {key: abs(metrics[1][key] - metrics[comparison][key]) for key in metrics[1]
                         if isinstance(metrics[1][key], (int, float)) and isinstance(metrics[comparison][key], (int, float))}
        report.update(samples=len(selected), sample_ids=[reader.items[i]['sample_id'] for i in selected],
                      labels=results[1]['labels'], tumour_probabilities_batch1=one.tolist(),
                      comparison_batch=comparison, tumour_probabilities_comparison=four.tolist(),
                      seconds=timings,
                      estimated_full_validation_seconds={k: v * 29164 / len(selected) for k, v in timings.items()},
                      max_absolute_probability_difference=float(delta.max()),
                      mean_absolute_probability_difference=float(delta.mean()),
                      differing_probability_count=int(np.count_nonzero(delta)),
                      class_predictions_identical=bool(np.array_equal(binary_decision(one), binary_decision(four))),
                      metrics=metrics, metric_absolute_differences=metric_deltas, memory_bytes=snapshots,
                      memory_scope='Fresh process; epoch9 Adam slots/accumulation buffers allocated; no backward graph/workspaces')
        assert report['class_predictions_identical']
        np.testing.assert_allclose(one, four, atol=1e-5, rtol=1e-5)
        assert max(metric_deltas.values()) <= 1e-6
        assert r.weight_hashes(model.weights) == saved['model_tensor_sha256']
        assert int(optimizer.iterations.numpy()) == 0 and accumulator.count == 0
        report.update(status='PASS', restored_weights_unchanged=True, new_stage_optimizer_iterations=0,
                      timestamp=r.utc_now())
    except BaseException as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        assert before == {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in files}
        assert json_before == {str(p): r.sha256_file(p) for p in files if p.suffix == '.json'}
        assert cache_before == {p: r.sha256_file(Path(p)) for p in cache_before}
        assert not fresh.exists()
        report.update(original_run_unchanged=True, cache_metadata_unchanged=True,
                      VAL2_access_attempts=len(blocked), resume_output_absent=str(fresh))
        output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'samples', 'max_absolute_probability_difference',
                     'mean_absolute_probability_difference', 'class_predictions_identical', 'old_adam_collected',
                     'memory_bytes', 'strict_checkpoint_restore', 'next_epoch', 'original_run_unchanged')}, indent=2))


if __name__ == '__main__':
    main()
