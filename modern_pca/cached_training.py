"""Cache-backed trainer integration and bounded Stage 2G dry run.

Production is reachable only through train_reimplementation --train-production.
This module's CLI runs one disposable optimizer update, never an epoch.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time

import numpy as np

from . import full_cache as f
from . import preprocessing_cache as c
from . import reimplementation as r


def checked_reader(root):
    reader = f.FullCacheReader(root)
    verification = f.read(Path(root) / 'verification.json')
    if verification['status'] != 'PASS' or verification['cache_catalog_sha256'] != r.sha256_file(Path(root) / 'cache.json'):
        raise ValueError('Full-cache verification is missing or stale')
    if f.digest(f.environment_contract()) != f.digest(reader.context['context']['contract']):
        raise ValueError('Cached preprocessing environment mismatch')
    for path, expected in reader.context['context']['protected_sha256'].items():
        r.sha256_file(Path(path), expected)
    return reader


def batches(reader, split, epoch, start=0, limit=None):
    indices = reader.indices(split, epoch if split == 'train' else None)
    if start < 0 or start > len(indices) or (split == 'train' and start % 100):
        raise ValueError('Resume requires a completed logical-batch boundary')
    indices = indices[start:] if limit is None else indices[start:start + limit]
    if not indices:
        return iter(())
    return c.dataset(reader, indices, batch_size=4, parallel=1, prefetch=2,
                     augment=split == 'train', epoch=epoch)


def train_batches(accumulator, dataset):
    loss_sum = 0.; samples = 0; updates = 0
    for images, labels, indices in dataset:
        size = int(images.shape[0])
        loss_sum += accumulator.add(images, labels) * size
        samples += size
        if accumulator.count == 100:
            accumulator.flush(); updates += 1
        if samples % 1000 == 0:
            r.LOG.info('Cached training processed %d samples', samples)
    if accumulator.count:
        accumulator.flush(); updates += 1
    if not samples or not np.isfinite(loss_sum):
        raise ValueError('Empty/nonfinite training result')
    return dict(samples=samples, loss=loss_sum / samples, updates=updates)


def validation(model, dataset):
    labels_out, scores, indices_out = [], [], []
    for images, labels, indices in dataset:
        values = model(images, training=False).numpy()
        if values.shape != (len(labels), 2) or not np.isfinite(values).all():
            raise ValueError('Nonfinite/invalid validation prediction')
        labels_out.extend(labels.numpy().tolist()); scores.extend(values[:, 1].tolist())
        indices_out.extend(indices.numpy().tolist())
    return dict(labels=labels_out, scores=scores, indices=indices_out)


def checkpoint_contract(cache_sha):
    return dict(cache_catalog_sha256=cache_sha, physical_batch=4, effective_batch=100,
                schedule=r.SCHEDULE, precision='float32', TF32=False, BN='frozen',
                model_parameters=r.PARAMETERS, split_sha256=c.SPLIT_HASHES)


def save_checkpoint(model, optimizer, accumulator, output, state, cache_sha):
    import tensorflow as tf
    if accumulator.count:
        raise ValueError('Cannot checkpoint incomplete gradient accumulation')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint.save_counter
    checkpoint.write(str(output / 'state'))
    value = dict(state=state, contract=checkpoint_contract(cache_sha),
                 optimizer_iterations=int(optimizer.iterations.numpy()),
                 files_sha256={p.name: r.sha256_file(p) for p in output.glob('state.*')},
                 model_tensor_sha256=r.weight_hashes(model.weights),
                 optimizer_tensor_sha256=r.weight_hashes(optimizer.variables),
                 completed_at_utc=r.utc_now())
    f.atomic_json(output / 'checkpoint.json', value)
    return value


def restore_checkpoint(model, optimizer, output, cache_sha, purpose):
    import tensorflow as tf
    output = Path(output); value = f.read(output / 'checkpoint.json')
    if f.digest(value['contract']) != f.digest(checkpoint_contract(cache_sha)) or value['state']['purpose'] != purpose:
        raise ValueError('Checkpoint cache/configuration/purpose mismatch')
    for name, digest in value['files_sha256'].items():
        if Path(name).name != name or (output / name).is_symlink():
            raise ValueError('Unsafe checkpoint path')
        r.sha256_file(output / name, digest)
    optimizer.build(model.trainable_variables)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint.save_counter
    checkpoint.read(str(output / 'state')).assert_consumed()
    if r.weight_hashes(model.weights) != value['model_tensor_sha256'] or r.weight_hashes(optimizer.variables) != value['optimizer_tensor_sha256']:
        raise ValueError('Checkpoint model/Adam state changed')
    if int(optimizer.iterations.numpy()) != value['optimizer_iterations']:
        raise ValueError('Optimizer resume iteration mismatch')
    return value['state']


def portable_roundtrip(model, output, probe):
    """Exercise the final production export format with disposable weights."""
    import tensorflow as tf
    output = Path(output)
    if output.exists():
        raise ValueError('Refusing to overwrite a portable checkpoint')
    expected = model(probe, training=False).numpy()
    hashes = r.weight_hashes(model.weights)
    started = time.perf_counter(); model.save(output)
    save_seconds = time.perf_counter() - started
    loaded = tf.keras.models.load_model(output, compile=False)
    if loaded.count_params() != model.count_params() or r.weight_hashes(loaded.weights) != hashes:
        raise ValueError('Portable checkpoint weights changed')
    if not np.array_equal(expected, loaded(probe, training=False).numpy()):
        raise ValueError('Portable checkpoint predictions changed')
    result = dict(status='PASS', file_sha256=r.sha256_file(output), bytes=output.stat().st_size,
                  save_seconds=save_seconds, save_reload_verify_seconds=time.perf_counter() - started,
                  weights_exact=True, prediction_exact=True, purpose='stage2g_disposable')
    del loaded; gc.collect()
    return result


def dry_run(cache, output):
    import tensorflow as tf
    from . import profile_reimplementation as e
    output = Path(output); r.ensure_output_separation(cache, [output]); r.logging_setup(output)
    r.LOG.info('Checking full cache and loading the disposable NASNetLarge model')
    started = time.perf_counter(); reader = checked_reader(cache)
    device = e.configure('gpu')
    model, counts = e.load_model(Path('runs/stage2e_profile'), r.SCHEDULE[0][0])
    optimizer = r.adam(r.SCHEDULE[0][2]); accumulator = r.Accumulator(model, optimizer)
    frozen = r.weight_hashes(model.non_trainable_variables)
    first = next(iter(batches(reader, 'train', 1, limit=4)))
    for image, label, index in zip(first[0].numpy(), first[1].numpy(), first[2].numpy()):
        item = reader.items[int(index)]
        expected = np.asarray(r.augment(c.to_float(reader.get(int(index))), item['sample_id'], 1))
        if not np.array_equal(image, expected) or label != item['ground_truth_code']:
            raise ValueError('Training cache scaling/flip/order mismatch')
    if not np.isfinite(model(first[0], training=False).numpy()).all():
        raise ValueError('Nonfinite forward pass')
    r.LOG.info('Dry run: 25 physical batches, one logical optimizer update')
    result = train_batches(accumulator, batches(reader, 'train', 1, limit=100))
    if result['samples'] != 100 or result['updates'] != 1 or int(optimizer.iterations.numpy()) != 1:
        raise ValueError('Dry run must perform exactly one 100-sample optimizer update')
    val_indices = reader.indices('internal_validation')
    # Both labels, selected deterministically from the actual validation split.
    selected = [next(i for i in val_indices if reader.items[i]['class_name'] == label) for label in r.CLASSES]
    selected += [i for i in sorted(val_indices, key=lambda i: r.rank_id(reader.items[i]['sample_id'], 'stage2g-dry-validation|42|')) if i not in selected][:30]
    r.LOG.info('Dry run: validation on 32 internal-validation samples')
    val = validation(model, c.dataset(reader, selected, batch_size=4, augment=False, prefetch=2))
    if val['indices'] != selected or val['labels'] != [reader.items[i]['ground_truth_code'] for i in selected]:
        raise ValueError('Validation ordering/labels mismatch')
    for images, _, indices in c.dataset(reader, selected[:4], batch_size=4, augment=False):
        for image, i in zip(images.numpy(), indices.numpy()):
            if not np.array_equal(image, c.to_float(reader.get(int(i)))):
                raise ValueError('Validation was augmented or incorrectly scaled')
    if frozen != r.weight_hashes(model.non_trainable_variables):
        raise ValueError('Frozen backbone/BN state changed')
    if any(not np.isfinite(v.numpy()).all() for v in model.weights):
        raise ValueError('Nonfinite model weights after update')
    cache_sha = r.sha256_file(Path(cache) / 'cache.json')
    state = dict(purpose='stage2g_disposable', epoch=1, next_offset=100, boundary=r.SCHEDULE[0][0])
    probe = first[0][:1]; expected = model(probe, training=False).numpy()
    r.LOG.info('Writing and verifying model/Adam checkpoint')
    checkpoint_started = time.perf_counter()
    saved = save_checkpoint(model, optimizer, accumulator, output / 'checkpoint', state, cache_sha)
    checkpoint_seconds = time.perf_counter() - checkpoint_started
    model.trainable_variables[-1].assign_add(tf.ones_like(model.trainable_variables[-1]))
    optimizer.iterations.assign_add(5)
    r.LOG.info('Restoring model, Adam slots, iteration and sample cursor')
    restore_started = time.perf_counter()
    restored = restore_checkpoint(model, optimizer, output / 'checkpoint', cache_sha, 'stage2g_disposable')
    restore_seconds = time.perf_counter() - restore_started
    if restored != state or not np.array_equal(expected, model(probe, training=False).numpy()):
        raise ValueError('Checkpoint prediction/cursor restore failed')
    resumed = next(iter(batches(reader, 'train', restored['epoch'], start=restored['next_offset'], limit=4)))
    if resumed[2].numpy().tolist() != reader.indices('train', 1)[100:104]:
        raise ValueError('Resume repeated or skipped samples')
    r.LOG.info('Testing disposable portable .keras export and reload')
    del accumulator, optimizer; gc.collect()
    portable = portable_roundtrip(model, output / 'disposable.keras', probe)
    report = dict(status='PASS', device=device, physical_batch=4, effective_batch=100,
                  model_parameters=model.count_params(), trainable_stage_counts=counts,
                  training=result, validation_samples=len(selected), validation_finite=True,
                  frozen_weights_exact=True, cache_scaling_exact=True, training_flips_exact=True,
                  validation_unaugmented_exact=True, checkpoint_restore_exact=True,
                  weights_finite=True, checkpoint_write_and_hash_seconds=checkpoint_seconds,
                  checkpoint_restore_and_verify_seconds=restore_seconds,
                  resume_cursor_exact=True, checkpoint=saved, seconds=time.perf_counter() - started,
                  portable_checkpoint=portable,
                  initial_checkpoint_sha256=r.sha256_file(Path('runs/stage2e_profile/initial.weights.h5')),
                  optimizer_updates_total=1, production_epoch_started=False, VAL2_accessed=False,
                  source_images_accessed=False, schedule=r.SCHEDULE)
    f.atomic_json(output / 'result.json', report)
    r.LOG.info('Disposable cache dry run PASS; exactly one optimizer update')
    return report


def production(args, train, validation_rows, summary):
    """Future explicit production entry; same schedule, loss, class mapping and flips."""
    from .evaluate_paper import calculate_binary_metrics
    reader = checked_reader(args.cache); device = r.gpu_setup()
    if args.batch_size != 4:
        raise ValueError('Verified cached production requires physical batch 4')
    model, counts = r.build_model(); cache_sha = r.sha256_file(Path(args.cache) / 'cache.json')
    meta = r.metadata(summary, 'production_epoch14', 4)
    meta.update(device=device, stage_trainable_parameters=counts, cache_catalog_sha256=cache_sha, completed_epochs=0)
    history = []; epoch = 0; resume = getattr(args, 'resume_cache_checkpoint', None)
    completed = 0
    if resume:
        checkpoint = f.read(Path(resume) / 'checkpoint.json')
        if checkpoint['state']['purpose'] != 'production':
            raise ValueError('Cannot resume production from a diagnostic checkpoint')
        completed = checkpoint['state']['completed_epochs']
        history = checkpoint['state']['history']
    for boundary, epochs, rate in r.SCHEDULE:
        end_epoch = epoch + epochs
        if completed >= end_epoch:
            epoch = end_epoch; continue
        r.configure_stage(model, boundary); optimizer = r.adam(rate)
        if resume:
            state = f.read(Path(resume) / 'checkpoint.json')['state']
            # Restore at the saved stage before moving into the next stage.
            if state['boundary'] != boundary:
                r.configure_stage(model, state['boundary'])
                previous_optimizer = r.adam(state['rate'])
                restore_checkpoint(model, previous_optimizer, resume, cache_sha, 'production')
                del previous_optimizer
                r.configure_stage(model, boundary)
            else:
                restore_checkpoint(model, optimizer, resume, cache_sha, 'production')
            resume = None
        accumulator = r.Accumulator(model, optimizer)
        for current in range(max(epoch, completed) + 1, end_epoch + 1):
            result = train_batches(accumulator, batches(reader, 'train', current))
            val = validation(model, batches(reader, 'internal_validation', current))
            if result['samples'] != f.COUNTS['train'] or len(val['labels']) != f.COUNTS['internal_validation']:
                raise ValueError('Production epoch sample count mismatch')
            history.append(dict(epoch=current, boundary=boundary, loss=result['loss'],
                                internal_validation=calculate_binary_metrics(val['labels'], val['scores'])))
            state = dict(purpose='production', completed_epochs=current, boundary=boundary, rate=rate, history=history)
            save_checkpoint(model, optimizer, accumulator, args.output / f'epoch_{current:02d}_resume', state, cache_sha)
            f.atomic_json(args.output / 'history.json', dict(epochs=history, selection='none; fixed epoch14'))
        epoch = end_epoch
        del accumulator, optimizer; gc.collect()
    if completed == 14:
        raise ValueError('Checkpoint already completed all 14 epochs; no training to resume')
    checkpoint = args.output / 'epoch14.keras'; model.save(checkpoint)
    meta.update(completed_epochs=14, model_sha256=r.sha256_file(checkpoint), completed_at_utc=r.utc_now())
    f.atomic_json(args.output / 'metadata.json', meta)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='Stage 2G disposable cache dry run: exactly one optimizer update')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    dry_run(args.cache, args.output)


if __name__ == '__main__':
    main()
