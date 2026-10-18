"""ENS model 2: multi-session training, internal validation and the ENS combination rule.

The frozen model-1 recipe (cached_training.production) at 500x500 input:
NASNetLarge ImageNet + Flatten + Dense(256, ReLU) + Dense(2, softmax); SCHEDULE;
Adam reset per stage and retained within epochs 1-7; frozen inference-mode BN;
float32 with TF32 off; physical batch 4 summed into effective batch 100 with the
final 55-sample group flushed; VAL1-epoch-v1 order and VAL1-flips-v1 flips;
fixed epoch 14, no selection or early stopping.

Only difference: checkpoints are also taken at logical-batch boundaries inside an
epoch (accumulator empty) and at a wall-clock deadline, so a 12 h Kaggle session
limit costs at most one checkpoint interval and a resume replays exactly the
remaining deterministic order with restored Adam state. VAL2 is never read.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import csv
import gc
import importlib.metadata
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path
import queue
import shutil
import sys
import threading
import time

import numpy as np

from . import ens_cache as e
from modern_pca import full_cache as f
from modern_pca import preprocessing_cache as c
from modern_pca import reimplementation as r
from modern_pca.evaluate_paper import calculate_binary_metrics, csv_bytes

PURPOSE = 'ens_model2'
CHECKPOINT = 'ens_checkpoint.json'
EPOCHS = sum(epochs for _, epochs, _ in r.SCHEDULE)
GREY_ZONE = (0.2, 0.8)  # Paper Fig. 2d trigger on the model-1 tumour probability.
PREDICTIONS = 'model2_internal_validation_predictions.csv'
MODEL = 'ens_model2_epoch14.keras'


def nasnet_grid(size):
    """NASNetLarge final feature-map side: valid 3x3 stride-2 stem, then four ceil-halving reductions."""
    side = (size - 3) // 2 + 1
    for _ in range(4):
        side = math.ceil(side / 2)
    return side


def head_parameters(size):
    return nasnet_grid(size) ** 2 * 4032 * 256 + 256 + 256 * 2 + 2


NASNET_BACKBONE = r.PARAMETERS - head_parameters(350)  # From the verified 350 px total.
PARAMETERS = NASNET_BACKBONE + head_parameters(e.SIZE)


def validate_model(model):
    import tensorflow as tf
    if tuple(model.input_shape) != (None, e.SIZE, e.SIZE, 3) or tuple(model.output_shape) != (None, 2):
        raise ValueError('Require 500x500 RGB input and two outputs')
    tail = model.layers[-3:]
    if not (isinstance(tail[0], tf.keras.layers.Flatten) and
            isinstance(tail[1], tf.keras.layers.Dense) and tail[1].units == 256 and
            tail[1].activation.__name__ == 'relu' and
            isinstance(tail[2], tf.keras.layers.Dense) and tail[2].units == 2 and
            tail[2].activation.__name__ == 'softmax' and model.count_params() == PARAMETERS):
        raise ValueError(f'Invalid model-2 NASNetLarge head/parameter count {model.count_params()} != {PARAMETERS}')


def build_model(weights='imagenet'):
    import tensorflow as tf
    tf.keras.mixed_precision.set_global_policy('float32')
    from modern_pca.models import build_classifier
    model = build_classifier(2, image_size=e.SIZE, weights=weights)
    validate_model(model)
    counts = {boundary: r.configure_stage(model, boundary) for boundary, _, _ in r.SCHEDULE}
    return model, counts


def stage_of(epoch):
    """(boundary, learning rate) of a 1-based epoch in the fixed schedule."""
    first = 1
    for boundary, epochs, rate in r.SCHEDULE:
        if first <= epoch < first + epochs:
            return boundary, rate
        first += epochs
    raise ValueError('Epoch outside the fixed 14-epoch schedule')


def rate_of(boundary):
    return next(rate for name, _, rate in r.SCHEDULE if name == boundary)


def prefetch(iterable, size=8):
    """Overlap cache reads with GPU compute; preserves order and re-raises loader errors."""
    items = queue.Queue(size); done = object()
    def work():
        try:
            for item in iterable:
                items.put(item)
            items.put(done)
        except BaseException as error:
            items.put(error)
    threading.Thread(target=work, daemon=True).start()
    while (item := items.get()) is not done:
        if isinstance(item, BaseException):
            raise item
        yield item


def batches(reader, indices, batch_size, epoch=None):
    """(images, labels, indices) in the given order; flips only when `epoch` is given (training)."""
    import tensorflow as tf
    from concurrent.futures import ThreadPoolExecutor
    def load():
        # Several tensors in flight at once: a single serial reader on NSCC's Lustre scratch gave
        # ~7 img/s against a 35 img/s GPU. Order is preserved; values are unchanged (SHA-checked).
        chunks = [indices[start:start + batch_size] for start in range(0, len(indices), batch_size)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            pending = []
            for chunk in chunks:
                pending.append((chunk, [pool.submit(reader.get, i) for i in chunk]))
                if len(pending) > 4:
                    chunk, futures = pending.pop(0)
                    yield [e.to_float(future.result()) for future in futures], chunk
            for chunk, futures in pending:
                yield [e.to_float(future.result()) for future in futures], chunk
    for images, chunk in prefetch(load()):
        if epoch is not None:
            images = tf.stack([r.augment(image, reader.items[i]['sample_id'], epoch) for image, i in zip(images, chunk)])
        else:
            # np.stack, not tf.stack: tf.stack on a list of NumPy arrays took ~0.3 s per image on Kaggle.
            images = tf.convert_to_tensor(np.stack(images))
        yield images, np.array([reader.items[i]['ground_truth_code'] for i in chunk], np.int64), chunk


def inference(model):
    """Compiled frozen forward pass; eager NASNet calls ran at ~1.3 img/s on a Kaggle T4."""
    import tensorflow as tf
    return tf.function(lambda images: model(images, training=False), reduce_retracing=True)


def initial_state():
    return dict(purpose=PURPOSE, epoch=1, phase='train', offset=0, loss_sum=0., updates=0, scores=[],
                boundary=stage_of(1)[0], history=[], final_scores=None)


def progress(state):
    return state['epoch'], state['phase'] == 'validation', state['offset']


def save(model, optimizer, accumulator, output, state, contract):
    """Write model + Adam state and the cursor; keep only the newest checkpoint in `output`."""
    import tensorflow as tf
    if accumulator is not None and accumulator.count:
        raise ValueError('Cannot checkpoint incomplete gradient accumulation')
    output = Path(output)
    directory = output / 'checkpoint_e{epoch:02d}_{phase}_{offset:06d}'.format(**state)
    if directory.exists() and not (directory / CHECKPOINT).exists():
        shutil.rmtree(directory)  # Uncommitted remnant of a session that died mid-write.
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint.save_counter
    checkpoint.write(str(directory / 'state'))
    f.atomic_json(directory / CHECKPOINT, dict(
        state=state, contract=contract, optimizer_iterations=int(optimizer.iterations.numpy()),
        files_sha256={p.name: r.sha256_file(p) for p in directory.glob('state.*')},
        model_tensor_sha256=r.weight_hashes(model.weights),
        optimizer_tensor_sha256=r.weight_hashes(optimizer.variables), completed_at_utc=r.utc_now()))
    for old in output.glob('checkpoint_e*'):
        if old.is_dir() and old != directory:
            shutil.rmtree(old)
    r.LOG.info('Checkpoint %s written', directory.name)
    return directory


def restore(model, optimizer, directory, contract):
    import tensorflow as tf
    directory = Path(directory); value = f.read(directory / CHECKPOINT)
    changed = sorted(k for k in set(contract) | set(value['contract'])
                     if f.digest(contract.get(k)) != f.digest(value['contract'].get(k)))
    if changed or value['state']['purpose'] != PURPOSE:
        raise ValueError(f'Checkpoint contract mismatch in {changed}; resume needs the same cache, batch, software and recipe')
    for name, digest in value['files_sha256'].items():
        if Path(name).name != name:
            raise ValueError('Unsafe checkpoint path')
        r.sha256_file(directory / name, digest)
    optimizer.build(model.trainable_variables)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint.save_counter
    checkpoint.read(str(directory / 'state')).assert_consumed()
    if (r.weight_hashes(model.weights) != value['model_tensor_sha256'] or
            r.weight_hashes(optimizer.variables) != value['optimizer_tensor_sha256']):
        raise ValueError('Checkpoint model/Adam state changed')
    if int(optimizer.iterations.numpy()) != value['optimizer_iterations']:
        raise ValueError('Optimizer resume iteration mismatch')
    return value['state']


def latest_checkpoint(roots):
    found = [p.parent for root in roots if Path(root).exists() for p in Path(root).rglob(CHECKPOINT)]
    return max(found, key=lambda d: progress(f.read(d / CHECKPOINT)['state'])) if found else None


def train(model, reader, output, contract, *, batch_size=4, resume=None, stop_at=math.inf, every=1800.,
          clock=time.time, configure=r.configure_stage):
    """Advance to the end of epoch 14 or until `stop_at`; returns the last checkpointed state."""
    Path(output).mkdir(parents=True, exist_ok=True)
    train_size = len(reader.indices('train'))
    validation = reader.indices('internal_validation')
    labels = [reader.items[i]['ground_truth_code'] for i in validation]
    predict = inference(model)
    optimizer = accumulator = None
    if resume is None:
        state = initial_state()
    else:
        saved_boundary = f.read(Path(resume) / CHECKPOINT)['state']['boundary']
        configure(model, saved_boundary); optimizer = r.adam(rate_of(saved_boundary))
        state = restore(model, optimizer, resume, contract)
        r.LOG.info('Resumed %s at epoch %d %s offset %d', resume, state['epoch'], state['phase'], state['offset'])
    saved, last = (progress(state) if resume else None), clock()
    started, done = time.perf_counter(), 0

    def boundary(force=False):
        """At a logical-batch boundary: checkpoint if due; True when this session must stop."""
        nonlocal saved, last
        now = clock(); stop = now >= stop_at
        if progress(state) != saved and (force or stop or now - last >= every):
            save(model, optimizer, accumulator, output, state, contract)
            saved, last = progress(state), now
        return stop

    def report(total):
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed else 0.
        r.LOG.info('Epoch %d %s %d/%d | %.2f img/s this session | epoch ETA %.2f h', state['epoch'], state['phase'],
                   state['offset'], total, rate, (total - state['offset']) / rate / 3600 if rate else float('nan'))

    while state['epoch'] <= EPOCHS:
        epoch = state['epoch']; boundary_name, rate = stage_of(epoch)
        if optimizer is None or state['boundary'] != boundary_name:
            # Adam resets at every stage; it is retained across epochs 1-7 and across sessions.
            optimizer = accumulator = None; gc.collect()
            configure(model, boundary_name); optimizer = r.adam(rate); state['boundary'] = boundary_name
        if accumulator is None:
            accumulator = r.Accumulator(model, optimizer)
        if state['phase'] == 'train':
            order = reader.indices('train', epoch)
            for images, targets, chunk in batches(reader, order[state['offset']:], batch_size, epoch):
                state['loss_sum'] += accumulator.add(images, targets) * len(chunk)
                state['offset'] += len(chunk); done += len(chunk)
                if accumulator.count == 100 or state['offset'] == len(order):
                    accumulator.flush(); state['updates'] += 1  # Includes the final partial group.
                    if state['offset'] % 2000 == 0:
                        report(len(order))
                    if boundary():
                        return state
            if state['offset'] != train_size:
                raise ValueError('Training epoch sample count mismatch')
            state.update(phase='validation', offset=0, scores=[])
        for images, _, chunk in batches(reader, validation[state['offset']:], batch_size):
            values = predict(images).numpy()
            if values.shape != (len(chunk), 2) or not np.isfinite(values).all():
                raise ValueError('Nonfinite/invalid validation prediction')
            state['scores'].extend(values[:, 1].tolist()); state['offset'] += len(chunk); done += len(chunk)
            if state['offset'] % 100 == 0 and state['offset'] < len(validation):
                if state['offset'] % 5000 == 0:
                    report(len(validation))
                if boundary():
                    return state
        if len(state['scores']) != len(validation):
            raise ValueError('Validation sample count mismatch')
        state['history'].append(dict(epoch=epoch, boundary=boundary_name, loss=state['loss_sum'] / train_size,
                                     updates=state['updates'],
                                     internal_validation=calculate_binary_metrics(labels, state['scores'])))
        r.LOG.info('Epoch %d done: loss %.6g, internal-validation accuracy %.4f', epoch,
                   state['history'][-1]['loss'], state['history'][-1]['internal_validation']['accuracy'])
        if epoch == EPOCHS:
            state['final_scores'] = state['scores']
        state.update(epoch=epoch + 1, phase='train', offset=0, loss_sum=0., updates=0, scores=[])
        f.atomic_json(Path(output) / 'history.json', dict(epochs=state['history'], selection='none; fixed epoch14'))
        if boundary(force=True):
            return state
    return state


def contract(reader, batch_size):
    import tensorflow as tf
    return dict(purpose=PURPOSE, cache_fingerprint=reader.fingerprint, input=list(e.SHAPE), parameters=PARAMETERS,
                physical_batch=batch_size, effective_batch=100, schedule=r.SCHEDULE, precision='float32', TF32=False,
                BN='frozen inference mode', order='VAL1-epoch-v1|42|<epoch>|', flips='VAL1-flips-v1|42|<epoch>|',
                split_sha256=c.SPLIT_HASHES, tensorflow=tf.__version__, keras=importlib.metadata.version('keras'))


def metadata(reader, batch_size, device, counts):
    split = json.loads(Path('manifests/VAL1_internal_v1/VAL1_split_summary.json').read_text())
    meta = r.metadata(split, PURPOSE, batch_size)
    meta.update(model=dict(backbone='NASNetLarge', weights='imagenet', input=list(e.SHAPE),
                           head=['Flatten', 'Dense(256,relu)', 'Dense(2,softmax)'], parameters=PARAMETERS),
                preprocessing=reader.context['preprocessing'], preprocessing_software=reader.context['software'],
                cache_fingerprint=reader.fingerprint, device=device, stage_trainable_parameters=counts,
                ens=dict(role='second ENS convnet (paper Fig. 2d)', grey_zone=list(GREY_ZONE),
                         magnification='whole 600 px x40 patch resized to 500 px (~x33; paper: approximately x35)'),
                deviations=r.DEVIATIONS + [
                    'Model-2 Supplementary Methods unavailable; model-1 constrained recipe reused at 500 px.',
                    'Intra-epoch checkpoints at logical-batch boundaries for 12 h Kaggle sessions.',
                    'Kaggle preprocessing/TensorFlow stack, not the Stage 2D WSL stack; no cross-environment pixel parity claimed.']
                + ([f'Physical batch {batch_size} instead of model 1\'s 4 (effective batch 100 unchanged): batch 4 ran out of '
                    'memory at 500 px on a 16 GB Kaggle T4 (deepest-stage peak 13.05 GB already at batch 2).']
                   if batch_size != 4 else []))
    return meta


def export(model, reader, output, state, meta):
    """Portable epoch-14 model plus its internal-validation predictions and checksum-bound sidecar."""
    from modern_pca import cached_training as t
    output = Path(output); validation = reader.indices('internal_validation')
    if state['epoch'] <= EPOCHS or len(state['final_scores'] or []) != len(validation):
        raise ValueError('Export requires the completed epoch-14 state')
    portable = t.portable_roundtrip(model, output / MODEL, e.to_float(reader.get(validation[0]))[None])
    portable['purpose'] = PURPOSE
    rows = [dict(sample_id=reader.items[i]['sample_id'], ground_truth_code=reader.items[i]['ground_truth_code'],
                 p_tumour=score) for i, score in zip(validation, state['final_scores'])]
    (output / PREDICTIONS).write_bytes(csv_bytes(rows, ['sample_id', 'ground_truth_code', 'p_tumour']))
    meta.update(purpose='ens_model2_epoch14', completed_epochs=EPOCHS, model_sha256=r.sha256_file(output / MODEL),
                predictions_sha256=r.sha256_file(output / PREDICTIONS), portable_checkpoint=portable,
                internal_validation=state['history'][-1]['internal_validation'], history=state['history'],
                completed_at_utc=r.utc_now())
    f.atomic_json(output / 'metadata.json', meta)
    return meta


def log_to(path):
    """Console + per-run file logging; safe to call again from the same notebook kernel."""
    root = logging.getLogger(); root.setLevel(logging.INFO)
    tags = {getattr(h, '_ens', None) for h in root.handlers}
    for tag, handler in (('console', logging.StreamHandler(sys.stdout)), (str(Path(path).resolve()), None)):
        if tag not in tags:
            handler = handler or logging.FileHandler(path, encoding='utf-8')
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
            handler._ens = tag; root.addHandler(handler)


def keras3():
    if int(importlib.metadata.version('keras').split('.')[0]) < 3:
        raise RuntimeError('Keras 3 (TensorFlow >= 2.16) is required, as in the Stage 2D environment')


def run_train(args):
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); log_to(output / 'train.log')
    keras3(); device = r.gpu_setup()
    reader = e.Model2Cache(args.cache_roots)
    resume = latest_checkpoint([*args.resume_roots, output])
    model, counts = build_model('imagenet' if resume is None else None)  # Resume overwrites every weight.
    terms = contract(reader, args.batch_size)
    meta = metadata(reader, args.batch_size, device, counts)
    r.LOG.info('Model 2: %d parameters; cache %s; resume from %s', model.count_params(), reader.fingerprint, resume)
    state = train(model, reader, output, terms, batch_size=args.batch_size, resume=resume,
                  stop_at=args.stop_at, every=args.checkpoint_minutes * 60)
    status = 'COMPLETE' if state['epoch'] > EPOCHS else 'PAUSED'
    if status == 'COMPLETE' and not (output / MODEL).exists():
        gc.collect(); export(model, reader, output, state, meta)
    f.atomic_json(output / 'status.json', dict(status=status, epoch=state['epoch'], phase=state['phase'],
                                               offset=state['offset'], resumed_from=str(resume), at_utc=r.utc_now()))
    r.LOG.info('Session %s at epoch %d %s offset %d', status, state['epoch'], state['phase'], state['offset'])
    return status


def run_probe(args):
    """Disposable: GPU peak memory and img/s at the first and deepest stages, plus a forecast."""
    import tensorflow as tf
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); log_to(output / 'probe.log')
    keras3(); device = r.gpu_setup(); reader = e.Model2Cache(args.cache_roots)
    model, counts = build_model()
    order = reader.indices('train', 1)[:200]; validation = reader.indices('internal_validation')[:100]
    stages = {}
    for boundary, _, rate in (r.SCHEDULE[0], r.SCHEDULE[-1]):
        r.configure_stage(model, boundary); optimizer = r.adam(rate); accumulator = r.Accumulator(model, optimizer)
        tf.config.experimental.reset_memory_stats('GPU:0')
        timings = []
        for group in (order[:100], order[100:]):  # First group includes tracing; the second is timed.
            begin = time.perf_counter()
            for images, targets, _ in batches(reader, group, args.batch_size, 1):
                accumulator.add(images, targets)
            accumulator.flush(); timings.append(time.perf_counter() - begin)
        predict = inference(model)
        predict(next(batches(reader, validation[:args.batch_size], args.batch_size))[0]).numpy()  # Trace, untimed.
        begin = time.perf_counter()
        for images, _, _ in batches(reader, validation, args.batch_size):
            predict(images).numpy()
        stages[boundary] = dict(train_images_per_second=100 / timings[1], validation_images_per_second=100 / (time.perf_counter() - begin),
                                peak_gpu_bytes=tf.config.experimental.get_memory_info('GPU:0')['peak'])
        r.LOG.info('Probe %s: %s', boundary, stages[boundary])
        optimizer = accumulator = None; gc.collect()
    # Per-image seconds interpolated linearly over the 8 stages between the measured endpoints.
    first, deepest = (stages[b] for b in (r.SCHEDULE[0][0], r.SCHEDULE[-1][0]))
    def epoch_seconds(weight):
        mix = lambda key: (1 - weight) / first[key] + weight / deepest[key]
        return f.COUNTS['train'] * mix('train_images_per_second') + f.COUNTS['internal_validation'] * mix('validation_images_per_second')
    weights = [0.] * 7 + [k / 7 for k in range(1, 8)]
    hours = [epoch_seconds(w) / 3600 for w in weights]
    result = dict(device=device, physical_batch=args.batch_size, stages=stages, epoch_hours=hours, total_hours=sum(hours),
                  scope='Disposable 1-update-per-stage probe; linear per-image interpolation between first and deepest stages; '
                        'excludes checkpoint IO and session startup.', VAL2_accessed=False)
    f.atomic_json(output / 'probe.json', result)
    r.LOG.info('Forecast: %.1f GPU hours for 14 epochs; per epoch %s', sum(hours), [round(h, 2) for h in hours])
    return result


def ens_combine(p_model1, p_model2, low=GREY_ZONE[0], high=GREY_ZONE[1]):
    """Paper Fig. 2d ENS: a model-1 tumour probability in [low, high] is re-evaluated by model 2,
    whose tumour probability then decides; outside the grey zone model 1 stands."""
    p1, p2 = np.asarray(p_model1, np.float64), np.asarray(p_model2, np.float64)
    both = np.concatenate([p1.ravel(), p2.ravel()])
    if p1.ndim != 1 or p1.shape != p2.shape or not np.isfinite(both).all() or (both < 0).any() or (both > 1).any():
        raise ValueError('Require aligned finite tumour probabilities in [0, 1]')
    triggered = (p1 >= low) & (p1 <= high)
    return np.where(triggered, p2, p1), triggered


_MODEL1 = None


def model1_init(paths, reference):
    global _MODEL1
    _MODEL1 = (e.ZipSource(paths), r.normalizers(Path(reference)))


def model1_inputs(rows):
    """Exact model-1 input (evaluate_paper.preprocess_legacy) from ZIP bytes."""
    source, normalizers = _MODEL1
    return np.stack([r.preprocess_legacy(row, e.Payload(source.read(row)), normalizers) for row in rows])


def ordered_map(pool, function, tasks, window):
    pending = deque()
    for task in tasks:
        pending.append(pool.submit(function, task))
        if len(pending) >= window:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def read_model2(directory, rows):
    directory = Path(directory); meta = f.read(directory / 'metadata.json')
    if meta.get('purpose') != 'ens_model2_epoch14' or meta.get('completed_epochs') != EPOCHS:
        raise ValueError('ENS needs the fixed epoch-14 model-2 output')
    r.sha256_file(directory / PREDICTIONS, meta['predictions_sha256'])
    with (directory / PREDICTIONS).open(encoding='utf-8', newline='') as stream:
        predictions = list(csv.DictReader(stream))
    if [p['sample_id'] for p in predictions] != [row['sample_id'] for row in rows] or any(
            int(p['ground_truth_code']) != row['ground_truth_code'] for p, row in zip(predictions, rows)):
        raise ValueError('Model-2 predictions do not match the frozen internal-validation split')
    return meta, [float(p['p_tumour']) for p in predictions]


def run_ensemble(args):
    """Internal-validation ENS: model 1 (epoch-14 checkpoint, 350 px) + model-2 predictions."""
    import tensorflow as tf
    from modern_pca.evaluate_reimplementation import probabilities_2, validate_sidecar
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False); log_to(output / 'ensemble.log')
    rows = [row for row in f.frozen_rows() if row['split'] == 'internal_validation']
    model2_meta, p2 = read_model2(args.model2, rows)
    sidecar = json.loads(Path(args.model1_metadata).read_text())
    validate_sidecar(sidecar, args.model1, False)
    if sidecar.get('purpose') != 'production_epoch14' or sidecar.get('completed_epochs') != 14:
        raise ValueError('ENS needs the fixed epoch-14 model-1 production checkpoint')
    tf.config.experimental.enable_tensor_float_32_execution(False)
    model = tf.keras.models.load_model(args.model1, compile=False); r.validate_model(model); model.trainable = False
    predict = inference(model)
    paths = e.download(args.archives, args.search)
    e.ZipSource(paths).check(rows)
    chunks = [rows[i:i + 100] for i in range(0, len(rows), 100)]; p1 = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'), initializer=model1_init,
                             initargs=({k: str(v) for k, v in paths.items()}, str(r.REFERENCE.resolve()))) as pool:
        for inputs in ordered_map(pool, model1_inputs, chunks, 2 * args.workers):
            for start in range(0, len(inputs), 4):
                batch = inputs[start:start + 4]
                p1.extend(probabilities_2(predict(batch).numpy(), len(batch))[:, 1].tolist())
            if len(p1) % 2000 == 0:
                r.LOG.info('Model 1 internal validation %d/%d', len(p1), len(rows))
    labels = [row['ground_truth_code'] for row in rows]
    final, triggered = ens_combine(p1, p2)
    table = [dict(sample_id=row['sample_id'], ground_truth_code=row['ground_truth_code'], p_tumour_model1=a,
                  p_tumour_model2=b, grey_zone_triggered=int(t), p_tumour_ens=y)
             for row, a, b, t, y in zip(rows, p1, p2, triggered, final)]
    (output / 'ens_internal_validation_predictions.csv').write_bytes(csv_bytes(table, list(table[0])))
    result = dict(split='VAL1 internal validation (patch-level; patient/slide leakage possible)', samples=len(rows),
                  rule='model-1 p_tumour in [0.2, 0.8] -> model-2 p_tumour decides; tumour iff p > 0.5 (ties benign)',
                  triggered=int(triggered.sum()), triggered_fraction=float(triggered.mean()),
                  paper_triggered_fraction=dict(VAL1=0.039, VAL2=0.036),
                  model1=calculate_binary_metrics(labels, p1), model2=calculate_binary_metrics(labels, p2),
                  ens=calculate_binary_metrics(labels, final),
                  model1_sha256=sidecar['model_sha256'], model2_sha256=model2_meta['model_sha256'],
                  completed_at_utc=r.utc_now(), VAL2_accessed=False)
    f.atomic_json(output / 'ens_metrics.json', result)
    r.LOG.info('ENS internal validation: model1 %.4f, model2 %.4f, ENS %.4f accuracy; %d triggered',
               result['model1']['accuracy'], result['model2']['accuracy'], result['ens']['accuracy'], result['triggered'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('probe', 'train'):
        command = sub.add_parser(name)
        command.add_argument('--cache-roots', type=Path, nargs='+', required=True)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--batch-size', type=int, choices=[1, 2, 4], default=4)
    train_command = sub.choices['train']
    train_command.add_argument('--resume-roots', type=Path, nargs='*', default=[])
    train_command.add_argument('--stop-at', type=float, required=True, help='Unix time to checkpoint and stop')
    train_command.add_argument('--checkpoint-minutes', type=float, default=30.)
    ensemble = sub.add_parser('ensemble')
    ensemble.add_argument('--model1', type=Path, required=True)
    ensemble.add_argument('--model1-metadata', type=Path, required=True)
    ensemble.add_argument('--model2', type=Path, required=True, help='Final model-2 output directory')
    ensemble.add_argument('--archives', type=Path, required=True)
    ensemble.add_argument('--search', type=Path, nargs='*', default=[])
    ensemble.add_argument('--output', type=Path, required=True)
    ensemble.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv)
    return {'probe': run_probe, 'train': run_train, 'ensemble': run_ensemble}[args.command](args)


if __name__ == '__main__':
    main()
