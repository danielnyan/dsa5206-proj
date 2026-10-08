"""Isolated bounded OOM investigation. Never calls the production trainer.

Each GPU candidate must run in its own process. All updates are disposable.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import weakref

import numpy as np

from modern_pca import cached_training as t, preprocessing_cache as c, reimplementation as r


class SerializedAverageAccumulator(r.Accumulator):
    """Candidate C only: reuse buffers for averaged gradients, one division at a time."""
    def _apply(self, count):
        import tensorflow as tf
        token = tf.constant(0.)
        for buffer in self.buffers:
            with tf.control_dependencies([token]):
                token = buffer.assign(buffer / tf.cast(count, tf.float32))
        with tf.control_dependencies([token]):
            self.optimizer.apply_gradients(list(zip(self.buffers, self.variables)))
        for buffer in self.buffers:
            buffer.assign(tf.zeros_like(buffer))
        return self.optimizer.iterations


def guard():
    attempts = []
    def audit(event, args):
        if event not in ('open', 'os.listdir', 'os.scandir') or not args:
            return
        if not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = str(Path(os.fsdecode(args[0])).absolute()).lower()
        if 'val2' in path or 'val_dataset_2' in path:
            attempts.append(path)
            raise RuntimeError('VAL2 access prohibited')
    sys.addaudithook(audit)
    return attempts


def inventory():
    result = {}
    for folder in ('runs/stage2h_production', 'runs/stage2h_production_resume_epoch08_vbatch2'):
        for path in Path(folder).rglob('*'):
            if path.is_file():
                result[str(path)] = dict(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                                        sha256=r.sha256_file(path) if path.suffix == '.json' else None)
    for name in ('cache.json', 'context.json', 'verification.json'):
        path = Path('runs/stage2g_full_cache/cache') / name
        result[str(path)] = dict(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                                sha256=r.sha256_file(path))
    return result


def fixture(candidate, output):
    """355 deterministic samples: three effective-100 updates plus final 55."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    def model():
        inputs = tf.keras.Input((16,))
        x = tf.keras.layers.Dense(12, activation='relu', trainable=False)(inputs)
        x = tf.keras.layers.BatchNormalization(trainable=False)(x, training=False)
        x = tf.keras.layers.Dense(9, activation='relu')(x)
        return tf.keras.Model(inputs, tf.keras.layers.Dense(2, activation='softmax')(x))
    a, b = model(), model(); b.set_weights(a.get_weights())
    rng = np.random.default_rng(42)
    x = rng.uniform(-1, 1, (355, 16)).astype('float32'); y = rng.integers(0, 2, 355)
    kind = SerializedAverageAccumulator if candidate == 'C' else r.Accumulator
    opt_a, opt_b = r.adam(1e-6), r.adam(1e-6)
    aa, bb = r.Accumulator(a, opt_a), kind(b, opt_b)
    frozen = r.weight_hashes(b.non_trainable_variables)
    gradients = []; steps = []
    max_gradient_abs = 0.; max_gradient_rel = 0.; max_weight_abs = 0.
    for start in range(0, 355, 100):
        end = min(start + 100, 355)
        for acc, physical in ((aa, 4), (bb, 2 if candidate == 'B' else 4)):
            for offset in range(start, end, physical):
                stop = min(offset + physical, end)
                loss = acc.add(x[offset:stop], y[offset:stop]); assert np.isfinite(loss)
            assert acc.count == end - start
        for av, bv in zip(aa.buffers, bb.buffers):
            av, bv = av.numpy() / (end-start), bv.numpy() / (end-start)
            assert np.isfinite(av).all() and np.isfinite(bv).all()
            delta = np.abs(av - bv)
            max_gradient_abs = max(max_gradient_abs, float(delta.max()))
            max_gradient_rel = max(max_gradient_rel, float((delta / np.maximum(np.abs(av), 1e-7)).max()))
            np.testing.assert_allclose(av, bv, atol=1e-6, rtol=1e-5)
            gradients.append(bv)
        aa.flush(); bb.flush()
        assert aa.count == bb.count == 0
        for av, bv in zip(a.weights, b.weights):
            max_weight_abs = max(max_weight_abs, float(np.abs(av.numpy()-bv.numpy()).max()))
            np.testing.assert_allclose(av.numpy(), bv.numpy(), atol=1e-7, rtol=1e-5)
        steps.append(end-start)
    expected = b(x[:16], training=False).numpy()
    predicted = a(x[:16], training=False).numpy()
    assert np.array_equal(predicted.argmax(1), expected.argmax(1))
    assert r.weight_hashes(b.non_trainable_variables) == frozen
    assert int(opt_a.iterations.numpy()) == int(opt_b.iterations.numpy()) == 4
    state = dict(purpose='epoch12_oom_fixture', completed_updates=4)
    t.save_checkpoint(b, opt_b, bb, output / 'fixture_checkpoint', state, 'f'*64)
    b.trainable_variables[-1].assign_add(tf.ones_like(b.trainable_variables[-1]))
    opt_b.iterations.assign_add(1)
    assert t.restore_checkpoint(b, opt_b, output / 'fixture_checkpoint', 'f'*64, 'epoch12_oom_fixture') == state
    np.testing.assert_array_equal(b(x[:16], training=False).numpy(), expected)
    np.savez(output / 'fixture_outputs.npz', predictions=expected,
             **{f'g{i}': v for i, v in enumerate(gradients)},
             **{f'w{i}': v.numpy() for i, v in enumerate(b.weights)})
    return dict(status='PASS', samples=355, logical_sizes=steps, updates=4,
                max_absolute_gradient_difference=max_gradient_abs,
                max_relative_gradient_difference=max_gradient_rel, relative_floor=1e-7,
                max_absolute_weight_difference=max_weight_abs,
                max_absolute_prediction_difference=float(np.abs(expected-predicted).max()),
                class_predictions_identical=True, frozen_unchanged=True,
                same_trainable_variable_shapes=[list(v.shape) for v in b.trainable_variables],
                checkpoint_roundtrip='PASS', tolerance_gradient=dict(atol=1e-6, rtol=1e-5),
                tolerance_weights=dict(atol=1e-7, rtol=1e-5))


def smoke(args, report):
    import tensorflow as tf
    cache = Path('runs/stage2g_full_cache/cache')
    resume = Path('runs/stage2h_production_resume_epoch08_vbatch2/epoch_11_resume')
    reader = t.checked_reader(cache)
    report['cache_compatibility'] = reader.environment_compatibility
    memory = lambda: tf.config.experimental.get_memory_info('GPU:0')
    boundary = {12: 'normal_conv_1_5', 13: 'normal_conv_1_3', 14: 'normal_conv_1_1'}[args.epoch]
    report['phase'] = 'build_restore'
    logging.info('Building model and restoring epoch 11')
    model, counts = r.build_model()
    # Same cross-stage resume sequence as production, including fresh optimizer.
    r.configure_stage(model, boundary); optimizer = r.adam(1e-6)
    r.configure_stage(model, 'normal_conv_1_7'); previous = r.adam(1e-6)
    state = t.restore_checkpoint(model, previous, resume, r.sha256_file(cache/'cache.json'), 'production')
    assert state['completed_epochs'] == 11 and state['boundary'] == 'normal_conv_1_7'
    assert state['rate'] == 1e-6 and state['validation_batch_size'] == 2
    assert [h['epoch'] for h in state['history']] == list(range(1, 12))
    assert t.f.read(resume.parent/'history.json')['epochs'] == state['history']
    report.update(restored_epoch=11, production_next_epoch=12, restored_iterations=int(previous.iterations.numpy()),
                  checkpoint_json_sha256=r.sha256_file(resume/'checkpoint.json'), strict_restore='PASS',
                  history_sha256=r.sha256_file(resume.parent/'history.json'), boundary=boundary,
                  trainable_parameters=counts[boundary], validation_batch=2, rate=1e-6)
    old_ref = weakref.ref(previous); del previous
    r.configure_stage(model, boundary)
    report['old_optimizer_live_after_del'] = old_ref() is not None
    assert int(optimizer.iterations.numpy()) == 0
    klass = SerializedAverageAccumulator if args.candidate == 'C' else r.Accumulator
    accumulator = klass(model, optimizer)
    report['variables'] = [dict(index=i, name=getattr(v, 'path', v.name), shape=list(v.shape),
                                bytes=int(np.prod(v.shape))*4) for i,v in enumerate(accumulator.variables)]
    report['memory_before_training'] = memory()
    physical = 2 if args.candidate == 'B' else 4
    report.update(physical_batch=physical, effective_batch=100, samples=0, microbatches=0,
                  optimizer_updates=0, logical_updates=[], fresh_adam_initial_iterations=0)
    sample_limit = args.updates*100 + (55 if args.chain else 0)
    expected_updates = args.updates + int(args.chain)
    indices = reader.indices('train', args.epoch)[:sample_limit]
    report['sample_ids'] = [reader.items[i]['sample_id'] for i in indices]
    assert len(indices) == len(set(indices)) == sample_limit
    frozen = r.weight_hashes(model.non_trainable_variables)
    tf.config.experimental.reset_memory_stats('GPU:0')
    started = time.perf_counter(); logical_start = started; loss_sum=0.
    for images, labels, ids in c.dataset(reader, indices, batch_size=physical, parallel=1, prefetch=2,
                                         augment=True, epoch=args.epoch):
        report['phase'] = 'microbatch'
        loss = accumulator.add(images, labels)
        report['samples'] += int(images.shape[0]); report['microbatches'] += 1
        loss_sum += loss*int(images.shape[0])
        if accumulator.count == 100:
            report['phase'] = 'optimizer_flush'
            report['accumulation_count_at_flush'] = accumulator.count
            report['memory_before_flush'] = memory()
            logging.info('Epoch %d %s: flush %d at sample %d, memory %s', args.epoch, args.candidate,
                         report['optimizer_updates']+1, report['samples'], memory())
            # Persist the cursor before an OOM/driver kill.
            t.f.atomic_json(args.output/'progress.json', report)
            accumulator.flush()
            report['optimizer_updates'] += 1
            report['logical_updates'].append(dict(samples=100, loss=loss_sum/100,
                seconds=time.perf_counter()-logical_start, memory=memory(), iterations=int(optimizer.iterations.numpy())))
            assert int(optimizer.iterations.numpy()) == report['optimizer_updates']
            loss_sum=0.; logical_start=time.perf_counter()
    if accumulator.count:
        report['phase'] = 'partial_optimizer_flush'
        assert args.chain and accumulator.count == 55
        report['accumulation_count_at_flush'] = accumulator.count
        accumulator.flush(); report['optimizer_updates'] += 1
        report['logical_updates'].append(dict(samples=55, loss=loss_sum/55,
            seconds=time.perf_counter()-logical_start, memory=memory(), iterations=int(optimizer.iterations.numpy())))
    report['training_seconds'] = time.perf_counter()-started
    report['images_per_second_including_compile'] = report['samples']/report['training_seconds']
    report['steady_images_per_second'] = (args.updates-1)*100/sum(x['seconds'] for x in report['logical_updates'][1:args.updates])
    assert report['samples'] == sample_limit and report['optimizer_updates'] == expected_updates
    assert accumulator.count == 0 and r.weight_hashes(model.non_trainable_variables) == frozen
    report.update(finite_losses_gradients=True, frozen_variables_unchanged=True, memory_after_training=memory(),
                  micro_tracing_count=accumulator.micro.experimental_get_tracing_count(),
                  apply_tracing_count=accumulator.apply.experimental_get_tracing_count())
    # Inspect the actual apply graph only after its natural first invocation.
    concrete = accumulator.apply.get_concrete_function(tf.TensorSpec([], tf.int32))
    report['apply_divisions'] = [dict(name=op.name, shape=op.outputs[0].shape.as_list())
                               for op in concrete.graph.get_operations() if op.type in ('RealDiv', 'Div')]
    del concrete  # Do not retain the old apply graph/resources across a stage transition.
    report['phase'] = 'validation'
    v = t.validation(model, t.batches(reader, 'internal_validation', args.epoch, limit=8, validation_batch_size=2))
    assert len(v['scores']) == 8
    report.update(validation_samples=8, validation_scores=v['scores'], memory_after_validation=memory(),
                  old_optimizer_live_after_updates=old_ref() is not None)
    report['model_weight_hashes_after'] = r.weight_hashes(model.weights)
    if args.chain:
        # Same process/model/allocator across remaining stages, with disposable
        # weights carried forward. No reinitialization or clear_session shortcut.
        report['phase'] = 'epoch12_disposable_checkpoint'
        checkpoint_state = dict(purpose='stage2h_oom_disposable', epoch=args.epoch, updates=expected_updates)
        t.save_checkpoint(model, optimizer, accumulator, args.output/'epoch12_disposable',
                          checkpoint_state, r.sha256_file(cache/'cache.json'))
        report['memory_after_disposable_checkpoint'] = memory()
        report['transition_stages'] = []
        for epoch, next_boundary in ((13, 'normal_conv_1_3'), (14, 'normal_conv_1_1')):
            old_acc_ref = weakref.ref(accumulator); old_opt_ref = weakref.ref(optimizer)
            before_collection = memory()
            del accumulator, optimizer
            gc.collect()
            lifetime = dict(before_collection=before_collection, after_collection=memory(),
                            old_accumulator_live=old_acc_ref() is not None,
                            old_optimizer_live=old_opt_ref() is not None)
            logging.info('Transition epoch %d lifetime: %s', epoch, lifetime)
            r.configure_stage(model, next_boundary); optimizer = r.adam(1e-6)
            accumulator = klass(model, optimizer)
            stage = dict(epoch=epoch, boundary=next_boundary, samples=0, updates=0,
                         microbatches=0, initial_iterations=int(optimizer.iterations.numpy()), logical_updates=[],
                         lifetime=lifetime)
            report['transition_stages'].append(stage)
            frozen = r.weight_hashes(model.non_trainable_variables)
            tf.config.experimental.reset_memory_stats('GPU:0')
            selected = reader.indices('train', epoch)[:sample_limit]
            assert len(selected) == len(set(selected)) == sample_limit
            started = time.perf_counter(); last=started
            for images, labels, ids in c.dataset(reader, selected, batch_size=physical,
                                                 parallel=1, prefetch=2, augment=True, epoch=epoch):
                report['phase'] = f'epoch{epoch}_microbatch'
                assert np.isfinite(accumulator.add(images, labels))
                stage['samples'] += len(labels); stage['microbatches'] += 1
                if accumulator.count == 100:
                    report['phase'] = f'epoch{epoch}_optimizer_flush'
                    logging.info('Transition epoch %d: flush %d, sample %d, memory %s',
                                 epoch, stage['updates']+1, stage['samples'], memory())
                    t.f.atomic_json(args.output/'progress.json', report)
                    accumulator.flush(); stage['updates'] += 1
                    stage['logical_updates'].append(dict(seconds=time.perf_counter()-last, memory=memory()))
                    last=time.perf_counter()
            if accumulator.count:
                report['phase'] = f'epoch{epoch}_partial_optimizer_flush'
                assert accumulator.count == 55
                accumulator.flush(); stage['updates'] += 1
                stage['partial_samples'] = 55
            stage['seconds']=time.perf_counter()-started
            assert stage['samples']==sample_limit and int(optimizer.iterations.numpy())==expected_updates
            assert accumulator.count==0 and frozen==r.weight_hashes(model.non_trainable_variables)
            stage['memory_after_training']=memory()
            stage['micro_tracing_count']=accumulator.micro.experimental_get_tracing_count()
            stage['apply_tracing_count']=accumulator.apply.experimental_get_tracing_count()
            val=t.validation(model,t.batches(reader,'internal_validation',epoch,limit=8,validation_batch_size=2))
            assert len(val['scores'])==8
            stage.update(status='PASS', memory_after_validation=memory(), frozen_unchanged=True,
                         steady_images_per_second=(args.updates-1)*100/sum(v['seconds'] for v in stage['logical_updates'][1:]))
            report['phase'] = f'epoch{epoch}_disposable_checkpoint'
            checkpoint_state = dict(purpose='stage2h_oom_disposable', epoch=epoch, updates=expected_updates)
            checkpoint_path = args.output/f'epoch{epoch}_disposable'
            t.save_checkpoint(model, optimizer, accumulator, checkpoint_path,
                              checkpoint_state, r.sha256_file(cache/'cache.json'))
            stage['memory_after_disposable_checkpoint'] = memory()
        report['phase'] = 'full_model_checkpoint_roundtrip'
        restored = t.restore_checkpoint(model, optimizer, checkpoint_path,
                                         r.sha256_file(cache/'cache.json'), 'stage2h_oom_disposable')
        assert restored == checkpoint_state and int(optimizer.iterations.numpy()) == expected_updates
        report['full_model_disposable_checkpoint_restore'] = 'PASS'
    report['phase'] = 'complete'; report['status'] = 'PASS'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=('baseline','A','B','C'), required=True)
    parser.add_argument('--epoch', type=int, choices=(12,13,14), default=12)
    parser.add_argument('--updates', type=int, choices=(3,4,5), default=3)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', action='store_true')
    parser.add_argument('--chain', action='store_true', help='After epoch12 smoke, retain process/model for epochs13 and14')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    if args.chain and (args.epoch != 12 or args.fixture):
        parser.error('--chain requires epoch12 model smoke')
    blocked = guard(); before=inventory()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    report=dict(status='STARTED', candidate=args.candidate, epoch=args.epoch, command=sys.argv,
                allocator=os.environ.get('TF_GPU_ALLOCATOR','default BFC'), production_training_started=False,
                scope='isolated disposable bounded test', started_at_utc=r.utc_now())
    try:
        import tensorflow as tf
        report['device']=r.gpu_setup()
        report['nvidia_smi']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free',
                                                     '--format=csv'],text=True)
        if args.fixture:
            report['fixture']=fixture(args.candidate,args.output); report['status']='PASS'
        else:
            smoke(args,report)
    except BaseException as error:
        report.update(status='OOM' if 'ResourceExhausted' in type(error).__name__ else 'FAIL',
                      error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        try: report['memory_at_failure']=tf.config.experimental.get_memory_info('GPU:0')
        except Exception: pass
        logging.exception('Bounded candidate failed')
    finally:
        report.update(finished_at_utc=r.utc_now(), VAL2_access_attempts=len(blocked),
                      original_artifacts_unchanged=before==inventory())
        if not report['original_artifacts_unchanged'] or blocked: report['status']='FAIL'
        t.f.atomic_json(args.output/'result.json',report)
    print(json.dumps({k:report.get(k) for k in ('status','candidate','epoch','samples','microbatches','optimizer_updates',
                      'phase','memory_after_training','memory_at_failure','steady_images_per_second')}),flush=True)
    return 0 if report['status']=='PASS' else 2


if __name__=='__main__':
    raise SystemExit(main())
