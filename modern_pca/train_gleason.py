"""Supervised legacy NASNetLarge training on both tumour-only datasets."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

from .gleason import (CLASSES, SCHEDULE, dataset, gpu_setup, read_rows, verify_cache,
                       provenance, write_json, sha)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifests', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--max-train-samples', type=int, default=96)
    p.add_argument('--resume', action='store_true', help='Resume last fully checkpointed stage')
    args = p.parse_args()
    if args.batch_size < 1 or args.max_train_samples < 3:
        p.error('Batch size must be positive and smoke sample limit at least 3')
    prepared = json.loads((args.manifests.parent/'prepared.json').read_text())
    if prepared['sample_only'] and not args.smoke:
        raise ValueError('Sample preparation cannot be used for production')
    train = sum([read_rows(args.manifests,d,'train') for d in ('crowd','sicap')],[])
    validation = sum([read_rows(args.manifests,d,'validation') for d in ('crowd','sicap')],[])
    if args.smoke:
        # Deterministic interleaving covers all represented classes and datasets.
        groups = [[r for r in train if r['dataset']==d and r['label']==k]
                  for d in ('crowd','sicap') for k in range(3)]
        import itertools
        train = [r for group in itertools.zip_longest(*groups) for r in group if r][:args.max_train_samples]
        validation = validation[:args.max_train_samples]
    if not train or not validation or {r['label'] for r in train} != {0,1,2}:
        raise ValueError('Need train classes GP3/GP4/GP5 and nonempty validation')
    verify_cache(args.cache, train+validation)
    settings = dict(manifests=prepared['manifests'], class_order=CLASSES,
                    batch_size=args.batch_size, seed=args.seed, smoke=args.smoke,
                    max_train_samples=args.max_train_samples if args.smoke else None,
                    initialization='imagenet', batchnorm='legacy trainable after stage boundary',
                    schedule=[SCHEDULE[0],SCHEDULE[-1]] if args.smoke else SCHEDULE,
                    preprocessing=prepared['preprocessing'], reference_sha256=prepared['reference_sha256'],
                    prenormalized_sources_accepted=prepared['prenormalized_sources_accepted'])
    start = 0
    history = []
    state = None
    if args.resume:
        state = json.loads((args.output/'checkpoint.json').read_text())
        if state['settings'] != json.loads(json.dumps(settings)):
            raise ValueError('Resume settings differ from checkpoint')
        checkpoint = args.output/state['checkpoint']
        if sha(checkpoint) != state['sha256']:
            raise ValueError('Checkpoint checksum mismatch')
        start,history = state['completed_stages'],state['history']
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    tf = gpu_setup(args.seed)
    if args.smoke:
        tf.debugging.enable_check_numerics()
    from .models import build_classifier, configure_fine_tuning
    if state:
        model = tf.keras.models.load_model(checkpoint, compile=False)
        model.backbone = next(layer for layer in model.layers if isinstance(layer,tf.keras.Model))
    else:
        # Transfer learning weights must already be staged when compute nodes
        # lack internet. Keras uses KERAS_HOME/models/nasnet_large_no_top.h5.
        weights = Path(os.environ.get('KERAS_HOME',Path.home()/'.keras'))/'models/nasnet_large_no_top.h5'
        if not weights.is_file():
            raise FileNotFoundError('Stage NASNet ImageNet weights with hpc/download_data.sh first')
        model = build_classifier(3, weights='imagenet', backbone_training=None)
    for boundary,_,_ in SCHEDULE:
        if boundary not in [l.name for l in model.backbone.layers]:
            raise ValueError(f'Missing stage boundary: {boundary}')
    run = dict(provenance(), **settings, complete=False, counts={'train':len(train),'validation':len(validation)},
               tensorflow=tf.__version__, parameters=model.count_params(),
               deviations=['replacement dataset labels and field of view', 'no stromal decontamination',
                           'no unreleased confidence-expansion training'],
               resume_semantics='completed-stage restart; interrupted stage is repeated; no bitwise continuation claim')
    if args.batch_size != 100:
        run['deviations'].append(f'physical batch {args.batch_size} instead of legacy 100')
    write_json(args.output/'run.json',run)
    class CheckFinite(tf.keras.callbacks.Callback):
        def on_train_batch_end(self,batch,logs=None):
            import math
            if not math.isfinite(float((logs or {}).get('loss',0))):
                raise FloatingPointError('Nonfinite training loss')
    # A fully checkpointed smoke run may only need final export/reload.
    train_ds = val_ds = result = None
    for stage,(boundary,epochs,lr) in enumerate(settings['schedule']):
        if stage < start:
            continue
        tf.keras.utils.set_random_seed(args.seed+stage)
        configure_fine_tuning(model,boundary)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='categorical_crossentropy',metrics=['accuracy'])
        train_ds = dataset(train,args.cache,args.batch_size,True,args.seed+stage)
        val_ds = dataset(validation,args.cache,args.batch_size)
        before = None
        frozen_before = None
        if args.smoke:
            before = model.layers[-1].get_weights()[0].copy()
            frozen = next((v for layer in model.backbone.layers if not layer.trainable
                           for v in layer.weights), None)
            if frozen is not None:
                frozen_before = frozen.numpy().copy()
        result = model.fit(train_ds,validation_data=val_ds,epochs=1 if args.smoke else epochs,
                           callbacks=[CheckFinite(),tf.keras.callbacks.CSVLogger(
                               str(args.output/f'stage-{stage}.csv'))],verbose=2)
        if args.smoke:
            import numpy as np
            if np.array_equal(before,model.layers[-1].get_weights()[0]):
                raise RuntimeError('Smoke run did not update classifier weights')
            if frozen_before is not None:
                np.testing.assert_array_equal(frozen_before,frozen.numpy())
        history.append(dict(stage=stage,boundary=boundary,history=result.history))
        # Two alternating files preserve the prior committed recovery point if
        # the scheduler kills the job during the next save.
        filename = f'checkpoint-{stage%2}.keras'
        temporary = args.output/f'pending-{stage%2}.keras'
        model.save(temporary)
        temporary.replace(args.output/filename)
        write_json(args.output/'checkpoint.json',dict(settings=settings,checkpoint=filename,
                   sha256=sha(args.output/filename),completed_stages=stage+1,history=history))
    model.save(args.output/'final.keras')
    if args.smoke:
        import numpy as np
        images,_ = next(iter(dataset(validation,args.cache,1)))
        expected = model(images,training=False).numpy()
        # Release the very large original model before restoring.
        del model, train_ds, val_ds, result
        tf.keras.backend.clear_session()
        import gc
        gc.collect()
        restored = tf.keras.models.load_model(args.output/'final.keras',compile=False)
        np.testing.assert_allclose(expected,restored(images,training=False).numpy(),rtol=1e-5,atol=1e-6)
    run.update(complete=True,model_sha256=sha(args.output/'final.keras'),history=history)
    write_json(args.output/'run.json',run)


if __name__ == '__main__':
    main()
