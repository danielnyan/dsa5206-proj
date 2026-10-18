"""Stage 2J secondary SICAPv2 transfer, 2xA40, one frozen epoch 15.

Separate from the original laptop Stage 2J and Vanda Stage 2H trainers.
Modes: --mode preflight (NO training updates) and --mode train (100 updates).
Requires an explicitly acknowledged 10x morphology deviation.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np
import tensorflow as tf
from . import reimplementation as r
from . import sicapv2 as s
from .vanda_distributed_training import distribute, make_train_fn, make_checkpoint, digest, write_json_atomic

MODEL_SHA = '433dd48aae800c659d1cba092af2859499bb804d2228ff978a6e9c9602b2adde'
MANIFEST_SHA = '52e87a6e814eb64c296493d0637b34872269f63d890b472d1c11d17914cd660d'
EXPECTED = 9959
BATCH = 100
BOUNDARY = 'normal_conv_1_1'
PURPOSE = 'secondary_SICAPv2_epoch15_vanda_2xa40'


def check_inputs(args):
    plan = json.loads(args.plan.read_text())
    if plan.get('status') not in ('GO', 'CONDITIONAL GO') or plan.get('preprocessing_failures') != 0:
        raise ValueError('Stage 2J local assessment is not approved')
    if plan.get('manifest_sha256') != MANIFEST_SHA or plan.get('samples') != EXPECTED:
        raise ValueError('Unexpected source manifest or sample count in plan')
    if plan.get('starting_model_sha256') == MODEL_SHA:
        raise ValueError('Unexpected plan: source plan must preserve laptop provenance')
    p = plan['protocol']
    required = {'starting_epoch':14, 'final_epoch':15, 'extra_epochs':1,
                'boundary':BOUNDARY, 'effective_batch':BATCH,
                'optimizer':'fresh Adam', 'learning_rate':1e-6,
                'epoch_for_flips':15, 'seed':42, 'precision':'float32',
                'BN':'frozen/inference', 'validation':'none; no checkpoint selection',
                'sampling':'each official training sample exactly once; no resampling',
                'class_order':['benign','tumour'], 'TF32':False,
                'loss':'unweighted sparse categorical crossentropy'}
    for k,v in required.items():
        if p.get(k) != v: raise ValueError(f'Frozen scientific plan mismatch: {k}')
    if p.get('physical_batch') != 4:
        raise ValueError('Source plan must remain frozen (physical batch 4)')
    if digest(args.manifest) != MANIFEST_SHA:
        raise ValueError('Selected manifest SHA mismatch')
    if digest(args.model) != MODEL_SHA:
        raise ValueError('Vanda epoch14 checkpoint export SHA mismatch')
    final_info = json.loads(args.model.with_name('final_model.json').read_text())
    if final_info.get('model_sha256') != MODEL_SHA or final_info.get('completed_epochs') != 14:
        raise ValueError('Vanda final_model.json mismatch')
    if digest(Path(r.REFERENCE)) != p['reference_sha256']:
        raise ValueError('Reference stain image changed')
    with args.manifest.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED or len({x['sample_id'] for x in rows}) != EXPECTED:
        raise ValueError('Selected rows duplicated or missing')
    for row in rows:
        if (row['partition'] != 'train' or row['official_label'] not in ('NC','G3','G4','G5')
                or row['sample_id'] != 'SICAPv2:' + row['filename']
                or Path(row['filename']).name != row['filename']
                or int(row['ground_truth_code']) != int(row['official_label'] != 'NC')):
            raise ValueError('Invalid SICAP row')
        row['ground_truth_code'] = int(row['ground_truth_code'])
    if rows != s.sample_order(rows):
        raise ValueError('Nonfrozen SICAP row ordering')
    if sum(x['ground_truth_code'] == 0 for x in rows) != 3773:
        raise ValueError('Benign count mismatch')
    root = args.images.resolve(strict=True)
    for i, row in enumerate(rows, 1):
        image = root / row['filename']
        if image.is_symlink() or not image.is_file() or not image.resolve().is_relative_to(root):
            raise ValueError(f'Missing or unsafe SICAP image: {image}')
        if args.verify_images and digest(image) != row['file_sha256']:
            raise ValueError(f'Image SHA mismatch: {image}')
        if args.verify_images and i % 2000 == 0:
            print(f'IMAGE VERIFIED {i}/{EXPECTED}', flush=True)
    return rows, plan


def gpu_setup():
    r.gpu_setup('float32')
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) != 2:
        raise RuntimeError(f'Require exactly two GPUs, found {len(gpus)}')
    strategy = tf.distribute.MirroredStrategy()
    if strategy.num_replicas_in_sync != 2:
        raise RuntimeError('Expected two replicas')
    return strategy


def load_model(strategy, path):
    with strategy.scope():
        model = tf.keras.models.load_model(path, compile=False)
        r.validate_model(model)
        backbones = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)
                     and any(x.name == BOUNDARY for x in layer.layers)]
        if len(backbones) != 1:
            raise RuntimeError('Cannot locate unique nested NASNet backbone')
        model.backbone = backbones[0]
        before = r.weight_hashes(model.weights)
        trainable_count = r.configure_stage(model, BOUNDARY)
        if before != r.weight_hashes(model.weights):
            raise RuntimeError('Stage configuration changed model weights')
        if any(x.trainable for x in model.backbone.layers
               if isinstance(x, tf.keras.layers.BatchNormalization)):
            raise RuntimeError('BatchNorm unfrozen')
    return model, trainable_count


def preprocess_batch(rows, root, reference):
    images = []
    for row in rows:
        path = root / row['filename']
        pixels = r.preprocess_legacy(row, path, reference)
        pixels = r.augment(pixels, row['sample_id'], 15)
        images.append(np.asarray(pixels, np.float32))
    x = np.stack(images)
    if x.shape != (len(rows),350,350,3) or not np.isfinite(x).all():
        raise ValueError(f'Invalid preprocessed tensor shape/values: {x.shape}')
    return x, np.asarray([row['ground_truth_code'] for row in rows], dtype=np.int32)


def contract(args, plan, model):
    return dict(purpose=PURPOSE, secondary=True, not_paper_post_val1_reproduction=True,
                magnification_deviation='SICAPv2 10x wide field accepted; resizing is not optical equivalence',
                source_plan_sha256=digest(args.plan), source_protocol=plan['protocol'],
                adapted_physical_batch_per_gpu=50, distributed_replicas=2,
                effective_batch=BATCH, starting_model_sha256=MODEL_SHA,
                selected_manifest_sha256=MANIFEST_SHA, sample_count=EXPECTED,
                optimizer='fresh Adam', optimizer_steps=100, lr=1e-6,
                trainable_boundary=BOUNDARY, epoch=15, validation='none',
                trainer_sha256=digest(Path(__file__)),
                core_sha256={n:digest(Path(__file__).with_name(n)) for n in
                   ('reimplementation.py','sicapv2.py','vanda_distributed_training.py')},
                tensorflow=tf.__version__, model_parameters=int(model.count_params()))


def train(args, rows, plan, strategy, model, trainable):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Training output must be absent or empty; no overwrite/restart')
    args.output.mkdir(parents=True, exist_ok=True)
    sig = contract(args, plan, model)
    write_json_atomic(args.output / 'contract.json', sig)
    with strategy.scope():
        optimizer = r.adam(1e-6)
        optimizer.build(model.trainable_variables)
    if int(optimizer.iterations.numpy()) != 0:
        raise RuntimeError('Adam not fresh')
    frozen = r.weight_hashes(model.non_trainable_variables)
    train_fn = make_train_fn(strategy, model, optimizer)
    reference = r.normalizers(r.REFERENCE)
    losses = 0.0
    count = 0
    started = time.perf_counter()
    for offset in range(0, EXPECTED, BATCH):
        group = rows[offset:offset+BATCH]
        x,y = preprocess_batch(group, args.images, reference)
        size = len(group)
        if size < 2: raise RuntimeError('Invalid final distributed batch')
        loss = float(train_fn(distribute(strategy,x,size),
                              distribute(strategy,y,size),
                              tf.constant(size,tf.int32)).numpy())
        if not np.isfinite(loss): raise RuntimeError('Nonfinite gradient-step loss')
        losses += loss * size
        count += size
        steps = int(optimizer.iterations.numpy())
        if steps % 10 == 0 or count == EXPECTED:
            print(f'SICAP TRAIN {count}/{EXPECTED} updates={steps}/100 '
                  f'loss={losses/count:.6f} elapsed_min={(time.perf_counter()-started)/60:.2f}',flush=True)
    if count != EXPECTED or int(optimizer.iterations.numpy()) != 100:
        raise RuntimeError('Training count/optimizer iteration mismatch')
    if frozen != r.weight_hashes(model.non_trainable_variables):
        raise RuntimeError('Frozen variables changed')
    checkpoint_dir = args.output / 'epoch_15_resume'
    temp_dir = args.output / ('.epoch15.' + uuid.uuid4().hex + '.staging')
    temp_dir.mkdir()
    ckpt = make_checkpoint(model, optimizer)
    ckpt.write(str(temp_dir/'state'))
    file_hashes = {f.name: digest(f) for f in temp_dir.glob('state.*')}
    metadata = dict(contract=sig, completed_epochs=15, training_samples=count,
                    optimizer_iterations=int(optimizer.iterations.numpy()),
                    model_tensor_sha256=r.weight_hashes(model.weights),
                    optimizer_tensor_sha256=r.weight_hashes(optimizer.variables),
                    files_sha256=file_hashes)
    write_json_atomic(temp_dir/'checkpoint.json',metadata)
    os.rename(temp_dir,checkpoint_dir)
    # Verify serialized checkpoint against its saved in-memory tensor hashes.
    # Avoid constructing a second NASNetLarge model (unnecessary GPU OOM risk).
    for name, sha in file_hashes.items():
        if digest(checkpoint_dir/name) != sha:
            raise RuntimeError('Checkpoint file checksum mismatch')
    restored = make_checkpoint(model, optimizer)
    restored.read(str(checkpoint_dir/'state')).assert_consumed()
    if (r.weight_hashes(model.weights) != metadata['model_tensor_sha256'] or
        r.weight_hashes(optimizer.variables) != metadata['optimizer_tensor_sha256'] or
        int(optimizer.iterations.numpy()) != 100):
        raise RuntimeError('Checkpoint restore parity failed')
    export = args.output / 'epoch15.keras'
    model.save(export)
    write_json_atomic(args.output/'history.json', dict(purpose=PURPOSE,
        epochs=[dict(epoch=15, samples=count, updates=100,
                     loss=losses/count,elapsed_seconds=round(time.perf_counter()-started,2))]))
    write_json_atomic(args.output/'metadata.json',dict(purpose=PURPOSE, completed_epochs=15,
        parent_model_sha256=MODEL_SHA,model_sha256=digest(export),
        contract_sha256=digest(args.output/'contract.json'),
        VAL2_images_accessed=False))
    write_json_atomic(args.output/'run_complete.json',dict(status='PASS',
        artifacts={p.name:digest(p) for p in args.output.iterdir() if p.is_file() and p.name!='run_complete.json'}))
    print('=== SICAP VANDA EPOCH15 TRAINING PASS ===',flush=True)



def train_smoke(args, rows, strategy, model):
    """One real distributed Adam update; never writes a checkpoint."""
    print("TRAIN-SMOKE START: one update, 50 images per GPU", flush=True)

    with strategy.scope():
        optimizer = r.adam(1e-6)
        optimizer.build(model.trainable_variables)

    assert int(optimizer.iterations.numpy()) == 0

    train_fn = make_train_fn(strategy, model, optimizer)
    reference = r.normalizers(r.REFERENCE)

    x, y = preprocess_batch(rows[:100], args.images, reference)

    loss = float(
        train_fn(
            distribute(strategy, x, 100),
            distribute(strategy, y, 100),
            tf.constant(100, tf.int32),
        ).numpy()
    )

    if not np.isfinite(loss):
        raise RuntimeError("Nonfinite training loss")

    if int(optimizer.iterations.numpy()) != 1:
        raise RuntimeError("Expected exactly one Adam update")

    print(
        f"TRAIN-SMOKE PASS: 100 samples, "
        f"GPU0=50 GPU1=50, Adam updates=1, loss={loss:.6f}",
        flush=True,
    )

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',required=True, choices=('preflight','train-smoke','train'))
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--accept-magnification-deviation', action='store_true', required=True)
    p.add_argument('--verify-images',action='store_true')
    args=p.parse_args()
    # Path guards: never read VAL2; no implicit access to its manifest/data.
    for path in (args.model,args.manifest,args.images,args.plan,args.output):
        if 'val2' in str(path).lower() or 'val_dataset_2' in str(path).lower():
            raise ValueError('VAL2 paths forbidden during epoch15')
    if args.output.resolve().is_relative_to(args.images.resolve()) or args.images.resolve().is_relative_to(args.output.resolve()):
        raise ValueError('Output overlaps image source')
    rows, plan = check_inputs(args)
    strategy=gpu_setup()
    model, trainable=load_model(strategy,args.model)
    print(f'VANDA SICAP mode={args.mode} replicas=2 trainable={trainable} '
          f'images={len(rows)} parent_sha={MODEL_SHA}',flush=True)
    if args.mode=='preflight':
        reference=r.normalizers(r.REFERENCE)
        # 100 distinct samples, no optimizer; same data path as training.
        x,y=preprocess_batch(rows[:100],args.images,reference)
        px=distribute(strategy,x,100)
        @tf.function(reduce_retracing=True)
        def preflight_step(distributed_images):
            per_replica = strategy.run(
                lambda images: model(images, training=False),
                args=(distributed_images,)
            )
            return strategy.gather(per_replica, axis=0)

        result = preflight_step(px)
        pred=strategy.gather(result,axis=0).numpy()
        if pred.shape != (100,2) or not np.isfinite(pred).all():
            raise RuntimeError('Preflight predictions invalid')
        print('PREFLIGHT PASS 100 samples, GPU0=50 GPU1=50, NO WEIGHT UPDATES',flush=True)
        return
    if args.mode == 'train-smoke':
        train_smoke(args, rows, strategy, model)
        return
    train(args,rows,plan,strategy,model,trainable)

if __name__=='__main__':
    main()
