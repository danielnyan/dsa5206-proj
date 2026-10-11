"""Independent Vanda 2xA40 NASNetLarge trainer, original Stage 2H files untouched.

Usage:
 python -m modern_pca.vanda_distributed_training --cache CACHE --output OUT --max-epochs 1
 python -m modern_pca.vanda_distributed_training --cache CACHE --output OUT --max-epochs 14

Checkpointing: one atomic, verified checkpoint after each FULL training +
internal-validation epoch. Interruptions during an epoch restart that epoch.
"""
import argparse
import gc
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import preprocessing_cache as c
from . import reimplementation as r
from .models import build_classifier
from .vanda_cache_consumer import open_verified_cache
from .evaluate_paper import calculate_binary_metrics

EXPECTED_TRAIN = 116655
EXPECTED_VAL = 29164
GLOBAL_BATCH = 100
VAL_BATCH = 64
PURPOSE = 'vanda_2xa40_production_epoch14'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json_atomic(path, value):
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    with open(tmp, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def stage_for(epoch):
    n = 0
    for index, (boundary, duration, rate) in enumerate(r.SCHEDULE):
        n += duration
        if epoch <= n:
            return index, boundary, rate
    raise ValueError(f'Epoch outside schedule: {epoch}')


def initial_model(strategy):
    with strategy.scope():
        model = build_classifier(
            num_classes=2, architecture='nasnetlarge',
            image_size=350, head='original', weights='imagenet'
        )
        r.validate_model(model)
    return model


def configure_optimizer(strategy, model, boundary, rate):
    # Stage 2H freezes BN and resets Adam whenever boundary changes.
    with strategy.scope():
        count = r.configure_stage(model, boundary)
        optimizer = r.adam(rate)
        optimizer.build(model.trainable_variables)
    return optimizer, count


def make_checkpoint(model, optimizer):
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    # Stage 2H uses .write()/.read(); materialize the save_counter first.
    ckpt.save_counter
    return ckpt


def signature(cache, model):
    imagenet = Path.home() / '.keras/models/nasnet_large_no_top.h5'
    if not imagenet.is_file():
        raise FileNotFoundError(f'Missing ImageNet weights: {imagenet}')
    return {
        'purpose': PURPOSE,
        'cache_sha256': digest(Path(cache) / 'cache.json'),
        'imagenet_sha256': digest(imagenet),
        'schedule': [list(row) for row in r.SCHEDULE],
        'model_parameters': int(model.count_params()),
        'global_batch': GLOBAL_BATCH,
        'validation_batch': VAL_BATCH,
        'train_count': EXPECTED_TRAIN,
        'validation_count': EXPECTED_VAL,
        'precision': 'float32',
        'tf32': False,
        'batchnorm': 'frozen',
        'seed': 42,
        'replicas': 2,
        'split_hashes': c.SPLIT_HASHES,
        'trainer_sha256': digest(Path(__file__)),
        'core_sha256': {name: digest(Path(__file__).with_name(name)) for name in
                        ('reimplementation.py', 'preprocessing_cache.py', 'models.py',
                         'vanda_cache_consumer.py')},
    }


def verify_checkpoint(folder, expected_signature):
    folder = Path(folder)
    if folder.is_symlink():
        raise ValueError(f'Symlink checkpoint rejected: {folder}')
    path = folder / 'manifest.json'
    if not path.is_file() or path.is_symlink():
        raise ValueError(f'Incomplete checkpoint: {folder}')
    meta = json.loads(path.read_text())
    if meta['signature'] != expected_signature:
        raise ValueError(f'Checkpoint contract mismatch: {folder}')
    epoch = int(meta['completed_epochs'])
    if not (1 <= epoch <= 14 and folder.name == f'epoch_{epoch:02d}_resume'):
        raise ValueError('Checkpoint epoch/folder mismatch')
    idx, boundary, rate = stage_for(epoch)
    if (meta['stage_index'] != idx or meta['boundary'] != boundary
            or meta['rate'] != rate or len(meta['history']) != epoch):
        raise ValueError(f'Checkpoint training stage/history mismatch: {folder}')
    files = meta['files_sha256']
    if not files or not any(n.endswith('.index') for n in files) or not any('.data-' in n for n in files):
        raise ValueError(f'Checkpoint shards/index missing: {folder}')
    for name, expected in files.items():
        file = folder / name
        if (Path(name).name != name or not file.is_file() or file.is_symlink()
                or digest(file) != expected):
            raise ValueError(f'Invalid checkpoint shard {name} in {folder}')
    return meta


def discover(output, expected_signature):
    history = []
    latest = None
    for epoch in range(1, 15):
        folder = output / f'epoch_{epoch:02d}_resume'
        if not folder.exists():
            if any((output / f'epoch_{n:02d}_resume').exists() for n in range(epoch + 1, 15)):
                raise ValueError('Non-contiguous checkpoints; manual investigation required')
            break
        latest = verify_checkpoint(folder, expected_signature)
        history = latest['history']
    return latest, history


def save_checkpoint(output, epoch, model, optimizer, history, sig):
    stage, boundary, rate = stage_for(epoch)
    destination = output / f'epoch_{epoch:02d}_resume'
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite completed checkpoint: {destination}')
    tmp = output / f'.epoch_{epoch:02d}_{uuid.uuid4().hex}.staging'
    tmp.mkdir()
    checkpoint = make_checkpoint(model, optimizer)
    checkpoint.write(str(tmp / 'state'))
    files = {p.name: digest(p) for p in sorted(tmp.glob('state.*')) if p.is_file()}
    if not any(n.endswith('.index') for n in files) or not any('.data-' in n for n in files):
        raise RuntimeError('Missing TensorFlow checkpoint shards')
    meta = {
        'signature': sig,
        'completed_epochs': epoch,
        'stage_index': stage,
        'boundary': boundary,
        'rate': rate,
        'history': history,
        'optimizer_iterations': int(optimizer.iterations.numpy()),
        'model_tensor_sha256': r.weight_hashes(model.weights),
        'optimizer_tensor_sha256': r.weight_hashes(optimizer.variables),
        'files_sha256': files,
    }
    write_json_atomic(tmp / 'manifest.json', meta)
    # Atomic publication of the completed checkpoint on the same filesystem.
    os.rename(tmp, destination)
    return meta


def restore_checkpoint(output, meta, model, optimizer, sig):
    folder = output / f"epoch_{meta['completed_epochs']:02d}_resume"
    verify_checkpoint(folder, sig)
    ckpt = make_checkpoint(model, optimizer)
    ckpt.read(str(folder / 'state')).assert_consumed()
    if (r.weight_hashes(model.weights) != meta['model_tensor_sha256']
            or r.weight_hashes(optimizer.variables) != meta['optimizer_tensor_sha256']
            or int(optimizer.iterations.numpy()) != meta['optimizer_iterations']):
        raise ValueError('Restored NASNet/Adam tensors differ from saved checkpoint')
    print(f"CHECKPOINT RESTORED epoch={meta['completed_epochs']} stage={meta['boundary']} updates={meta['optimizer_iterations']}", flush=True)


def distribute(strategy, values, n):
    left = (n + 1) // 2
    def to_replica(ctx):
        if ctx.replica_id_in_sync_group == 0:
            return tf.convert_to_tensor(values[:left])
        return tf.convert_to_tensor(values[left:n])
    return strategy.experimental_distribute_values_from_function(to_replica)


def make_train_fn(strategy, model, optimizer):
    @tf.function(reduce_retracing=True)
    def step(dx, dy, n):
        def replica(x, y):
            with tf.GradientTape() as tape:
                probabilities = model(x, training=False)
                losses = tf.keras.losses.sparse_categorical_crossentropy(y, probabilities)
                loss = tf.reduce_sum(losses) / tf.cast(n, tf.float32)
            variables = model.trainable_variables
            gradients = tape.gradient(loss, variables)
            if any(g is None for g in gradients):
                raise RuntimeError('Disconnected gradient')
            tf.debugging.assert_all_finite(loss, 'Nonfinite training loss')
            for gradient in gradients:
                tf.debugging.assert_all_finite(gradient, 'Nonfinite gradient')
            optimizer.apply_gradients(zip(gradients, variables))
            return loss
        losses = strategy.run(replica, args=(dx, dy))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)
    return step


def train_epoch(reader, epoch, strategy, model, optimizer, train_fn, log_every):
    indices = reader.indices('train', epoch)
    if len(indices) != EXPECTED_TRAIN or len(set(indices)) != EXPECTED_TRAIN:
        raise ValueError('Training index count or uniqueness mismatch')
    ds = c.dataset(reader, indices, batch_size=GLOBAL_BATCH, parallel=1,
                   prefetch=1, augment=True, epoch=epoch)
    sample_total = 0
    weighted_loss = 0.
    updates = 0
    initial_iter = int(optimizer.iterations.numpy())
    start = time.perf_counter()
    for images, labels, original_indices in ds:
        count = int(labels.shape[0])
        if count < 2 or count > GLOBAL_BATCH:
            raise ValueError(f'Invalid distributed group: {count}')
        # Keep identity and order of all indices; no padding or repeat.
        observed = original_indices.numpy().tolist()
        if observed != indices[sample_total:sample_total + count]:
            raise ValueError('Data-loader shuffled/duplicated indices')
        x = images.numpy()
        y = labels.numpy()
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite input image')
        loss = float(train_fn(distribute(strategy, x, count),
                              distribute(strategy, y, count),
                              tf.constant(count, tf.int32)).numpy())
        if not np.isfinite(loss):
            raise ValueError('Nonfinite loss')
        weighted_loss += loss * count
        sample_total += count
        updates += 1
        if updates % log_every == 0 or sample_total == EXPECTED_TRAIN:
            elapsed = time.perf_counter() - start
            print(f'TRAIN epoch={epoch} samples={sample_total}/{EXPECTED_TRAIN} updates={updates} elapsed_min={elapsed/60:.2f} loss_to_date={weighted_loss/sample_total:.6f}', flush=True)
    expected_updates = (EXPECTED_TRAIN + GLOBAL_BATCH - 1) // GLOBAL_BATCH
    if (sample_total != EXPECTED_TRAIN or updates != expected_updates
            or int(optimizer.iterations.numpy()) != initial_iter + expected_updates):
        raise RuntimeError('Unexpected sample/update count for epoch')
    return {'samples': sample_total, 'updates': updates,
            'loss': weighted_loss / sample_total,
            'elapsed_seconds': round(time.perf_counter() - start, 2)}


def make_validation_fn(strategy, model):
    @tf.function(reduce_retracing=True)
    def step(dx):
        predictions = strategy.run(lambda x: model(x, training=False), args=(dx,))
        return strategy.gather(predictions, axis=0)
    return step


def validate_epoch(reader, epoch, strategy, predict_fn):
    indices = reader.indices('internal_validation')
    if len(indices) != EXPECTED_VAL or len(set(indices)) != EXPECTED_VAL:
        raise ValueError('Validation indices mismatch')
    ds = c.dataset(reader, indices, batch_size=VAL_BATCH, parallel=1,
                   prefetch=1, augment=False, epoch=epoch)
    labels_all, scores_all = [], []
    count = 0
    started = time.perf_counter()
    for images, labels, original_indices in ds:
        n = int(labels.shape[0])
        if original_indices.numpy().tolist() != indices[count:count + n]:
            raise ValueError('Validation sample index mismatch')
        probs = predict_fn(distribute(strategy, images.numpy(), n)).numpy()
        if probs.shape != (n, 2) or not np.isfinite(probs).all():
            raise ValueError('Invalid validation probabilities')
        labels_all.extend(labels.numpy().tolist())
        scores_all.extend(probs[:, 1].tolist())
        count += n
    if count != EXPECTED_VAL:
        raise ValueError('Incomplete internal validation')
    metrics = calculate_binary_metrics(labels_all, scores_all)
    # Normalize NumPy scalar outputs before writing JSON.
    metrics = json.loads(json.dumps(metrics, default=lambda o: o.item() if isinstance(o, np.generic) else o.tolist()))
    return {'samples': count, 'metrics': metrics,
            'elapsed_seconds': round(time.perf_counter() - started, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--max-epochs', type=int, default=14, choices=range(1, 15))
    ap.add_argument('--log-every', type=int, default=50)
    args = ap.parse_args()
    if args.log_every < 1:
        raise ValueError('log-every must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    r.gpu_setup('float32')
    if len(tf.config.list_physical_devices('GPU')) != 2:
        raise RuntimeError('Production requires exactly two TensorFlow GPUs')
    strategy = tf.distribute.MirroredStrategy()
    if strategy.num_replicas_in_sync != 2:
        raise RuntimeError('Expected two MirroredStrategy replicas')
    reader = open_verified_cache(args.cache)
    model = initial_model(strategy)
    sig = signature(args.cache, model)
    latest, history = discover(output, sig)
    completed = int(latest['completed_epochs']) if latest else 0
    if completed >= args.max_epochs and args.max_epochs != 14:
        print(f'ALREADY COMPLETE through epoch {completed}', flush=True)
        return
    if completed == 14:
        stage, boundary, rate = stage_for(14)
        optimizer, _ = configure_optimizer(strategy, model, boundary, rate)
        restore_checkpoint(output, latest, model, optimizer, sig)
        export = output / 'epoch14.keras'
        if not export.exists():
            model.save(export)
            write_json_atomic(output / 'final_model.json',
                              {'model_sha256': digest(export), 'completed_epochs': 14,
                               'purpose': PURPOSE, 'checkpoint': 'epoch_14_resume'})
        print('ALL 14 EPOCHS COMPLETE', flush=True)
        return
    start_epoch = completed + 1
    optimizer = None
    active_stage = None
    if latest:
        # Restore optimizer with *saved* trainability, even when the next
        # epoch begins a different fine-tuning stage.
        active_stage = latest['stage_index']
        _, saved_boundary, saved_rate = stage_for(completed)
        optimizer, _ = configure_optimizer(strategy, model, saved_boundary, saved_rate)
        restore_checkpoint(output, latest, model, optimizer, sig)
    print(f'PRODUCTION START epoch={start_epoch} end={args.max_epochs} replicas=2', flush=True)
    for epoch in range(start_epoch, args.max_epochs + 1):
        stage, boundary, rate = stage_for(epoch)
        if active_stage != stage:
            if optimizer is not None:
                del optimizer
                gc.collect()
            optimizer, trainable = configure_optimizer(strategy, model, boundary, rate)
            active_stage = stage
            print(f'STAGE epoch={epoch} boundary={boundary} rate={rate} trainable={trainable} ADAM_RESET', flush=True)
        train_fn = make_train_fn(strategy, model, optimizer)
        val_fn = make_validation_fn(strategy, model)
        training = train_epoch(reader, epoch, strategy, model, optimizer, train_fn, args.log_every)
        validation = validate_epoch(reader, epoch, strategy, val_fn)
        row = {'epoch': epoch, 'boundary': boundary, 'rate': rate,
               'training': training, 'internal_validation': validation}
        history.append(row)
        meta = save_checkpoint(output, epoch, model, optimizer, history, sig)
        print(f'EPOCH CHECKPOINT COMMITTED epoch={epoch} path={output / f"epoch_{epoch:02d}_resume"} iterations={meta["optimizer_iterations"]}', flush=True)
        write_json_atomic(output / 'history.json', {'purpose': PURPOSE, 'history': history, 'completed_epochs': epoch})
        del train_fn, val_fn
        gc.collect()
    if args.max_epochs == 14:
        export = output / 'epoch14.keras'
        if not export.exists():
            model.save(export)
            write_json_atomic(output / 'final_model.json',
                              {'model_sha256': digest(export), 'completed_epochs': 14,
                               'purpose': PURPOSE, 'checkpoint': 'epoch_14_resume'})
        print('ALL 14 EPOCHS COMPLETE; epoch14.keras exported', flush=True)
    else:
        print(f'STOPPED SAFELY at requested --max-epochs {args.max_epochs}', flush=True)


if __name__ == '__main__':
    main()
